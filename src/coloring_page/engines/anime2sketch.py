"""Optional pretrained-model conversion engine (Anime2Sketch).

Requires the ``ml`` extra (``pip install -e ".[ml]"``) for ``torch``, and a
manually downloaded pretrained weights file -- neither is required by, or
bundled with, the rest of this project. See the README's "Extending with
an ML engine" section for setup instructions and license information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

from coloring_page.drawing import Drawing, paths_from_mask
from coloring_page.engines._anime2sketch_arch import UnetGenerator, build_generator
from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.exceptions import WeightsMissingError
from coloring_page.postprocess import (
    gradient_edges,
    hysteresis_centerline,
    normalize_percentile,
)
from coloring_page.weights import resolve_weights_path, verify_checksum

#: Environment variable used to point at a local weights file, in place of
#: the default lookup locations below. Deprecated in favor of the shared
#: ``COLORING_PAGE_WEIGHTS_DIR`` (see coloring_page.weights), kept as a
#: higher-priority override for this file only.
WEIGHTS_ENV_VAR = "COLORING_PAGE_ANIME2SKETCH_WEIGHTS"

#: Checkpoint filename within the resolved weights directory.
_WEIGHTS_FILENAME = "anime2sketch.pth"

#: Where weights are looked for when neither a constructor argument, an
#: env var, nor ``COLORING_PAGE_WEIGHTS_DIR`` is set: a `weights/` folder
#: relative to the current working directory (matching the upstream
#: Anime2Sketch project's own convention) first, then the per-user cache
#: directory.
DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "coloring_page" / _WEIGHTS_FILENAME
_LOCAL_WEIGHTS_PATH = Path("weights") / _WEIGHTS_FILENAME


def _resolve_weights_path(explicit_path: str | Path | None) -> Path:
    """Pick the weights file to use -- see ``coloring_page.weights.resolve_weights_path``."""
    return resolve_weights_path(
        _WEIGHTS_FILENAME,
        explicit_path=explicit_path,
        legacy_env_var=WEIGHTS_ENV_VAR,
        local_path=_LOCAL_WEIGHTS_PATH,
    )


def _resize_and_pad(rgb: np.ndarray, load_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize preserving aspect ratio, then reflect-pad to a multiple of 256.

    The network has 8 downsampling stages, so both spatial dimensions
    fed into it must be multiples of 256. Resizing directly to a
    ``load_size x load_size`` square (the previous approach) satisfies
    that but distorts non-square photos, squashing them before
    inference and stretching the output back afterwards. Scaling the
    longest side to ``load_size`` and padding the shorter side up to the
    next multiple of 256 keeps the network's input square-multiple
    requirement without changing the image's proportions. Reflection
    padding (rather than a solid color) avoids introducing a hard,
    artificial edge along the padded border that the network could
    otherwise render as a spurious line.

    Parameters
    ----------
    rgb : np.ndarray
        RGB image, shape ``(H, W, 3)``.
    load_size : int
        Target size, in pixels, for the image's longest side.

    Returns
    -------
    tuple[np.ndarray, tuple[int, int]]
        The padded image, and the ``(height, width)`` of the resized
        (pre-padding) content -- needed to crop the padding back off the
        network's output before the final resize to the original size.
    """
    height, width = rgb.shape[:2]
    scale = load_size / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    pad_bottom = (-new_height) % 256
    pad_right = (-new_width) % 256
    padded = cv2.copyMakeBorder(resized, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    return padded, (new_height, new_width)


_WEIGHTS_HELP = (
    "Anime2Sketch pretrained weights not found at {path}.\n"
    "This engine requires weights that this project does not bundle or "
    "download automatically:\n"
    "  1. Download the 'default' model weights (netG.pth) from the "
    "official Google Drive link in the Anime2Sketch README:\n"
    "     https://github.com/Mukosame/Anime2Sketch#download-pretrained-weights\n"
    f"  2. Save the file to {_LOCAL_WEIGHTS_PATH} (relative to the current "
    f"directory), {DEFAULT_WEIGHTS_PATH}, or set the {WEIGHTS_ENV_VAR} "
    "environment variable to wherever you saved it.\n"
    "Anime2Sketch is MIT-licensed (Copyright (c) 2021 Xiaoyu Xiang); see "
    "THIRD_PARTY_LICENSES.md."
)


class Anime2SketchEngine(ConversionEngine):
    """Extract line art using the pretrained Anime2Sketch U-Net model.

    Unlike the other engines in this package, this one is a pretrained
    neural network rather than a hand-tuned classical-CV filter. It was
    trained specifically to turn illustration/anime-style artwork into
    clean sketches, which tends to suit painted or drawn source images
    better than gradient-based edge detection (see ``cartoon.py`` and the
    other engines for the classical alternatives).
    """

    name = "anime2sketch"
    requires_serial_execution = True

    def __init__(
        self,
        weights_path: str | Path | None = None,
        load_size: int = 512,
        gamma: float = 1.6,
        postprocess_strategy: Literal["nms", "hysteresis"] = "hysteresis",
    ) -> None:
        """Store where to find the pretrained weights and inference options.

        Parameters
        ----------
        weights_path : str | Path | None, optional
            Path to the ``.pth`` weights file. If None, resolved via
            :func:`_resolve_weights_path` (env var, then
            ``./weights/anime2sketch.pth``, then the per-user cache dir).
        load_size : int, optional
            Target size (in pixels) for the image's longest side before
            being fed through the network, by default 512, matching the
            size the upstream pretrained weights were trained/tested
            with. The shorter side is scaled proportionally, then both
            sides are reflect-padded up to the next multiple of 256 (the
            network has 8 downsampling stages) -- see
            :func:`_resize_and_pad`. Must itself be a multiple of 256.
        gamma : float, optional
            Gamma-correction factor applied to the input image before
            inference, by default 1.6. On source images with large dark
            or shadowed regions, this pretrained network otherwise tends
            to collapse those regions into a solid black blob instead of
            linework; lifting shadow detail beforehand (`output = input **
            (1/gamma)`) avoids that failure mode. Set to 1.0 to disable.
        postprocess_strategy : {"nms", "hysteresis"}, optional
            Centerline-extraction strategy applied to the network's soft
            output before tracing it into vector paths, by default
            ``"hysteresis"``. A ``Drawing`` has no concept of soft,
            pencil-shaded output -- unlike this engine's pre-vector
            behavior, there is no way to opt out of binarization now, since
            vector line art is inherently already a binary decision about
            where a stroke is. This is a forced override, not a
            re-validated choice: this network's soft output is
            wide/blurry (~12% ink after thresholding), and skeletonizing
            that down to a proper 1px centerline collapses ink coverage
            to ~2% -- below this project's 3-7% admissibility band --
            which crashes recall far more than ``"nms"``
            (:func:`~coloring_page.postprocess.gradient_edges`) doubling
            every stroke into its two edges costs precision. Measured
            with ``scripts/compare.py``: switching from ``"nms"`` to
            ``"hysteresis"`` drops this engine's ``f1_normalized`` on all
            3 reference pairs (0.447/0.183/0.322 ->
            0.193/0.110/0.164). The default is set to ``"hysteresis"``
            anyway, because a doubled-edge output is not a centerline
            regardless of what it scores. Pass
            ``postprocess_strategy="nms"`` explicitly to restore the
            higher-scoring behavior. Revisit once
            ``hysteresis_centerline``'s thresholds are retuned for this
            network's output, or a sharper soft map is available.
        """
        self.weights_path = _resolve_weights_path(weights_path)
        self.load_size = load_size
        self.gamma = gamma
        self.postprocess_strategy = postprocess_strategy
        self._model: UnetGenerator | None = None

    def _get_model(self) -> UnetGenerator:
        """Lazily build the network and load its pretrained weights.

        Raises
        ------
        WeightsMissingError
            If no weights file exists at ``self.weights_path``, with
            instructions for obtaining one.
        WeightsChecksumError
            If the weights file exists but doesn't match its pinned
            SHA256 (only checked for filenames known to
            ``coloring_page.weights.CHECKSUMS``).
        """
        if self._model is not None:
            return self._model

        if not self.weights_path.exists():
            raise WeightsMissingError(_WEIGHTS_HELP.format(path=self.weights_path))
        verify_checksum(self.weights_path, _WEIGHTS_FILENAME)

        model = build_generator()
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
        model.eval()

        self._model = model
        return model

    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> Drawing:
        """Run the pretrained network and trace its output into vector line art.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        model = self._get_model()
        original_height, original_width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded, (content_height, content_width) = _resize_and_pad(rgb, self.load_size)
        if self.gamma != 1.0:
            padded = (255 * (padded / 255.0) ** (1.0 / self.gamma)).astype(np.uint8)
        tensor = torch.from_numpy(padded).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5  # [0, 1] -> [-1, 1]

        with torch.no_grad():
            output = model(tensor)

        # Network output is a single-channel Tanh activation in [-1, 1];
        # rescale to a standard [0, 255] grayscale line-art image.
        sketch = output.squeeze().clamp(-1, 1).cpu().numpy()
        sketch = ((sketch + 1) / 2 * 255).astype(np.uint8)
        # Crop the reflect-padding back off at network resolution before
        # resizing to the original size, so padded content never gets
        # blended into the final image.
        sketch = sketch[:content_height, :content_width]
        sketch = cv2.resize(
            sketch, (original_width, original_height), interpolation=cv2.INTER_CUBIC
        )
        if debug is not None:
            debug.save("raw_sketch", sketch)

        normalized = normalize_percentile(sketch)
        if self.postprocess_strategy == "hysteresis":
            centerline = hysteresis_centerline(normalized)
        elif self.postprocess_strategy == "nms":
            centerline = gradient_edges(normalized)
        else:
            raise ValueError(
                f"Unknown postprocess_strategy {self.postprocess_strategy!r}; "
                "expected 'nms' or 'hysteresis'."
            )

        paths = paths_from_mask(centerline, kind="detail")
        return Drawing(paths=paths, aspect_ratio=original_width / original_height)
