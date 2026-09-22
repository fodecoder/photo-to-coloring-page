"""Standalone raster detail-line conversion engine (``--style lineart-raster``).

Isolates exactly one stage of ``coloring_page.engines.lineart.LineArtEngine``
(that engine's Stage B: controlnet_aux's ``lineart_anime`` detail-line
network, see :func:`~coloring_page.engines.lineart._detail_lines`) as its
own engine, deliberately without SAM 2 segmentation, merging, pruning, or
vectorization. Two hands-on measurements against this repo motivated this:
Stage B's raw soft response is close to what a good reference line-art
image actually looks like (antialiased, ~4-5% full-black ink, ~10% midtone
gray), while this project's vector pipeline (binarize -> thin -> trace)
demonstrably rejects that same reference image once run through it -- see
``coloring_page.artwork`` for why. This engine measures Stage B alone,
kept raster, against that same standard.

Unlike ``lineart.py`` (``experimental = True``, not yet measured against
reference images with real weights), this engine is registered as
non-experimental: the whole point of adding it is to actually run and
measure it (see ``scripts/ablation.py``), not to gate it behind a flag
first.

Deliberately does not import anything from ``coloring_page.postprocess``:
no binarization, no thinning, no ``hysteresis_centerline``. The only
manipulation this engine performs on the raw detector response is an
optional gamma curve (:attr:`LineArtRasterEngine.contrast_gamma`, identity
by default) and an optional invert -- reintroducing any raster-destroying
step here would defeat the purpose of measuring what Stage B alone can do.

Requires the ``lineart_raster`` extra (``pip install -e ".[lineart_raster]"``)
for ``torch`` and ``controlnet_aux`` -- deliberately not the ``lineart``
extra's ``sam2``, since this engine has no segmentation stage. Requires the
``netG.pth`` checkpoint already used by ``lineart.py``'s Stage B; see
``scripts/fetch_weights.py`` and ``THIRD_PARTY_LICENSES.md`` (this
checkpoint's redistribution license is the same still-unconfirmed one
documented there for ``lineart.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

# Imported at module level (rather than only inside functions that use
# them) purely so importing this module raises ImportError whenever the
# `lineart_raster` extra's dependencies are missing -- matching
# lineart.py's own convention. registry.py's try/except ImportError around
# this module's import is what keeps "lineart-raster" out of `ENGINES`
# until `pip install -e ".[lineart_raster]"` has actually been run.
import controlnet_aux  # noqa: F401
import cv2
import numpy as np

from coloring_page.artwork import RasterArtwork
from coloring_page.engines.base import ConversionEngine, DebugSink

# Reused rather than reimplemented -- see lineart.py's own import of the
# same helper for why (matches controlnet_aux's own resize convention).
from coloring_page.engines.informative_drawings import _resize_short_side
from coloring_page.exceptions import WeightsMissingError
from coloring_page.weights import resolve_weights_path, verify_checksum

#: Checkpoint filename, shared with ``lineart.py``'s Stage B -- both
#: engines legitimately load the identical file.
_DETAIL_WEIGHTS_FILENAME = "netG.pth"

WEIGHTS_ENV_VAR = "COLORING_PAGE_LINEART_RASTER_WEIGHTS"

DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "coloring_page" / _DETAIL_WEIGHTS_FILENAME
_LOCAL_WEIGHTS_PATH = Path("weights") / _DETAIL_WEIGHTS_FILENAME

_WEIGHTS_HELP = (
    "lineart_anime detail-line checkpoint (netG.pth) not found at {path}.\n"
    "Run 'python scripts/fetch_weights.py --filename netG.pth --dest "
    f"{_LOCAL_WEIGHTS_PATH}' (requires the 'lineart_raster' extra's "
    "huggingface_hub dependency).\n"
    f"Save it to {_LOCAL_WEIGHTS_PATH} (relative to the current directory), "
    f"{DEFAULT_WEIGHTS_PATH}, or set the {WEIGHTS_ENV_VAR} environment "
    "variable to wherever you saved it.\n"
    "See THIRD_PARTY_LICENSES.md for this checkpoint's license."
)


def _resolve_weights_path(explicit_path: str | Path | None) -> Path:
    """Pick the weights file to use -- see ``coloring_page.weights.resolve_weights_path``."""
    return resolve_weights_path(
        _DETAIL_WEIGHTS_FILENAME,
        explicit_path=explicit_path,
        legacy_env_var=WEIGHTS_ENV_VAR,
        local_path=_LOCAL_WEIGHTS_PATH,
    )


def _resolve_device(device: Literal["auto", "cpu", "cuda", "mps"]) -> str:
    """Resolve ``"auto"`` to a concrete torch device string; pass the rest through.

    Deliberately a local copy of ``lineart.py``'s identical helper rather
    than an import from it: importing ``lineart.py`` at all would pull in
    its module-level ``import sam2``, defeating this module's entire
    reason to exist (see the module docstring).
    """
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(weights_path: Path, *, device: str) -> Any:
    """Lazily construct controlnet_aux's ``LineartAnimeDetector`` from a checkpoint on disk.

    Deliberately not imported from ``lineart.py``: that keeps this module
    free of a ``sam2`` dependency even though the ~15 lines of model
    construction are otherwise identical to ``lineart.py``'s own
    ``_build_detail_model``.
    """
    if not weights_path.exists():
        raise WeightsMissingError(_WEIGHTS_HELP.format(path=weights_path))
    verify_checksum(weights_path, _DETAIL_WEIGHTS_FILENAME)
    from controlnet_aux import LineartAnimeDetector
    from PIL import Image

    detector = LineartAnimeDetector.from_pretrained(str(weights_path.parent)).to(device)

    def _run(image_rgb: np.ndarray) -> np.ndarray:
        result = detector(Image.fromarray(image_rgb), output_type="np")
        return np.asarray(result)

    return _run


class LineArtRasterEngine(ConversionEngine):
    """Run controlnet_aux's ``lineart_anime`` detector and return its raw response, unmodified.

    A single stage, kept raster: resize to a working resolution -> run the
    detector -> resize back -> :class:`~coloring_page.artwork.RasterArtwork`.
    No binarization, thinning, region merging, or vectorization -- see the
    module docstring for why.
    """

    name = "lineart-raster"
    experimental = False
    #: Holds a model resident in (V)RAM across calls, same rationale as
    #: LineArtEngine.requires_serial_execution.
    requires_serial_execution = True

    def __init__(
        self,
        *,
        resolution: int = 1024,
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        invert: bool = False,
        contrast_gamma: float = 1.0,
        weights_path: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        """Store this engine's parameters.

        Parameters
        ----------
        resolution : int, optional
            Short-side working resolution the detector runs at, by
            default 1024. This is the parameter ``coloring_page.api``
            wires ``Profile.detail`` into via
            ``coloring_page.profile.RASTER_RESOLUTION_PRESETS`` --
            "simplify" for this engine is a resolution change, not a
            filter, since a lower-resolution pass has a proportionally
            larger receptive field and draws thicker, less detailed
            strokes.
        device : {"auto", "cpu", "cuda", "mps"}, optional
            Inference device, by default ``"auto"``.
        invert : bool, optional
            Flip the output (``255 - response``) a second time, on top of
            the unconditional polarity correction ``convert`` already
            applies to the detector's own raw (white-line-on-black)
            response -- by default False. Not needed in normal use; only
            useful if a future detector/checkpoint swap reintroduces the
            opposite (dark-line-on-white) convention and this correction
            would otherwise double-flip it.
        contrast_gamma : float, optional
            The only permitted manipulation of the detector's raw
            response: a gamma curve applied to the ink-positive signal
            (``response`` inverted, raised to ``contrast_gamma``,
            inverted back), by default 1.0 (identity, no-op). Values below
            1.0 push midtones darker (more ink), above 1.0 push them
            lighter -- never a threshold, so antialiasing survives at any
            setting.
        weights_path : str | Path | None, optional
            Explicit checkpoint path, overriding the default lookup (env
            var, then ``weights/``, then the per-user cache directory).
        model : Any | None, optional
            A pre-built ``image_rgb -> np.ndarray`` callable, injected in
            place of lazily building one from weights on first use. This
            is what lets tests substitute a stub without any real weights
            or GPU.
        """
        self.resolution = resolution
        self.device = device
        self.invert = invert
        self.contrast_gamma = contrast_gamma
        self.weights_path = weights_path
        self._model = model

    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> RasterArtwork:
        """Run the detector and return its response as a ``RasterArtwork``.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        height, width = image.shape[:2]

        model = self._model
        if model is None:
            weights = _resolve_weights_path(self.weights_path)
            model = _build_model(weights, device=_resolve_device(self.device))

        resized = _resize_short_side(image, self.resolution)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        response = np.asarray(model(image_rgb))
        if response.ndim == 3:
            response = cv2.cvtColor(response, cv2.COLOR_RGB2GRAY)
        response = response.astype(np.uint8)

        # LineartAnimeDetector's raw response is white lines on a black
        # background -- confirmed against real weights during this
        # engine's Phase 2 measurement (see scripts/ablation.py), the
        # opposite of this project's convention (0=ink/black,
        # 255=paper/white, see coloring_page.artwork.RasterArtwork). This
        # correction is unconditional, not part of the optional `invert`
        # flag below: it fixes the checkpoint's own polarity, it isn't a
        # style choice. `lineart.py`'s Stage B shares this same detector
        # call and, per its own docstring, was never exercised against
        # real weights before this -- it likely needs the same fix.
        response = 255 - response
        if debug is not None:
            debug.save("detail_response", response)

        if self.contrast_gamma != 1.0:
            ink_positive = 255.0 - response.astype(np.float64)
            curved = 255.0 * (ink_positive / 255.0) ** self.contrast_gamma
            response = np.clip(255.0 - curved, 0, 255).astype(np.uint8)
        if self.invert:
            response = 255 - response

        result = cv2.resize(response, (width, height), interpolation=cv2.INTER_LINEAR)
        return RasterArtwork(
            image=result,
            aspect_ratio=width / height,
            meta={"engine": self.name, "resolution": self.resolution},
        )
