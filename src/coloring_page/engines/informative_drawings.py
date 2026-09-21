"""Optional pretrained-model conversion engine (Informative Drawings).

Requires the ``ml`` extra (``pip install -e ".[ml]"``) for ``torch``, and a
manually downloaded pretrained weights file -- neither is required by, or
bundled with, the rest of this project. See the README's "Extending with
an ML engine" section for setup instructions and license information.

Recommended over ``anime2sketch`` for photographed/painted illustrations:
Anime2Sketch is trained on already-clean digital anime line art, not on
photos of paintings or printed illustrations, so its results shift and
distort geometry it wasn't trained to expect. Informative Drawings (Chan,
Durand, Isola, CVPR 2022) is trained specifically to produce the kind of
clean, geometry-preserving line drawing this project targets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import torch

from coloring_page.engines._informative_drawings_arch import Generator, build_generator
from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.postprocess import soft_map_to_line_art

#: Environment variable used to point at a local weights file, in place of
#: the default lookup locations below.
WEIGHTS_ENV_VAR = "COLORING_PAGE_INFORMATIVE_DRAWINGS_WEIGHTS"

#: Where weights are looked for when neither a constructor argument nor
#: WEIGHTS_ENV_VAR is set: a `weights/` folder relative to the current
#: working directory first, then a per-user cache directory.
DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "coloring_page" / "informative_drawings.pth"
_LOCAL_WEIGHTS_PATH = Path("weights") / "informative_drawings.pth"


def _resolve_weights_path(explicit_path: str | Path | None) -> Path:
    """Pick the weights file to use, in priority order.

    Priority: an explicit path (constructor argument) >
    ``COLORING_PAGE_INFORMATIVE_DRAWINGS_WEIGHTS`` > an existing
    ``./weights/informative_drawings.pth`` relative to the current
    working directory > the per-user cache directory (used as the final
    fallback even if it doesn't exist yet, so callers get a consistent
    "expected" path to report in error messages).
    """
    if explicit_path is not None:
        return Path(explicit_path)

    env_path = os.environ.get(WEIGHTS_ENV_VAR)
    if env_path:
        return Path(env_path)

    if _LOCAL_WEIGHTS_PATH.exists():
        return _LOCAL_WEIGHTS_PATH

    return DEFAULT_WEIGHTS_PATH


def _resize_short_side(image: np.ndarray, resolution: int, *, multiple: int = 64) -> np.ndarray:
    """Scale so the image's short side is ``resolution``, rounded to a multiple of ``multiple``.

    Matches ``controlnet_aux.util.resize_image`` verbatim (as of
    huggingface/controlnet_aux's ``master`` branch), which is what the
    redistributed ``lllyasviel/Annotators`` weights are exercised with by
    ControlNet's own "lineart" preprocessor::

        k = resolution / min(H, W)
        H, W = round(H * k / 64) * 64, round(W * k / 64) * 64
        interpolation = LANCZOS4 if k > 1 else AREA

    Each side is rounded to the nearest multiple of ``multiple``
    *independently*, which can shift the aspect ratio very slightly (a
    handful of pixels) -- an intentional, minor tradeoff upstream accepts
    to keep both dimensions compatible with the network's downsampling
    stages, not a bug in this reimplementation. A fully-convolutional
    generator has no fixed input size, but it does have a scale it was
    trained at: feeding it a squashed square (this engine's previous
    behavior) changes both the aspect ratio and the apparent scale of
    every object in the scene, which is a much larger distortion than
    this rounding.

    Parameters
    ----------
    image : np.ndarray
        Image to resize, shape ``(H, W, ...)``.
    resolution : int
        Target size, in pixels, for the image's shorter side before
        rounding.
    multiple : int, optional
        Round each output dimension to the nearest multiple of this
        value, by default 64 (the network's downsampling factor).

    Returns
    -------
    np.ndarray
        Resized copy of ``image``.
    """
    height, width = image.shape[:2]
    scale = resolution / min(height, width)
    new_height = max(multiple, round(height * scale / multiple) * multiple)
    new_width = max(multiple, round(width * scale / multiple) * multiple)
    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


_WEIGHTS_HELP = (
    "Informative Drawings pretrained weights not found at {path}.\n"
    "This engine requires weights that this project does not bundle:\n"
    "  Recommended: run 'python scripts/fetch_weights.py --filename "
    f"sk_model.pth --dest {_LOCAL_WEIGHTS_PATH}' (requires the 'ml' "
    'extra\'s huggingface_hub dependency). Downloads the "fine" detail '
    "checkpoint from the lllyasviel/Annotators Hugging Face mirror -- "
    'measured to score higher than the "coarse" alternative '
    "(sk_model2.pth) on 2 of this project's 3 reference images, see the "
    "commit that pinned this choice. See THIRD_PARTY_LICENSES.md for the "
    "license caveat this accepts.\n"
    "  Alternative (official source, manual): download 'model.zip' from "
    "the official Google Drive link in the Informative Drawings README "
    "(https://github.com/carolineec/informative-drawings#testing), unzip "
    "it, and locate checkpoints/contour_style/netG_A_latest.pth (or "
    "another style's netG_A_latest.pth -- contour_style is recommended "
    "for photographed/printed illustrations).\n"
    f"  Either way, save the file to {_LOCAL_WEIGHTS_PATH} (relative to "
    f"the current directory), {DEFAULT_WEIGHTS_PATH}, or set the "
    f"{WEIGHTS_ENV_VAR} environment variable to wherever you saved it.\n"
    "Informative Drawings is MIT-licensed (Copyright (c) 2022 Caroline "
    "Chan); see THIRD_PARTY_LICENSES.md."
)


class InformativeDrawingsEngine(ConversionEngine):
    """Extract line art using the pretrained Informative Drawings generator.

    Unlike the classical engines in this package, this one is a
    pretrained neural network. It was trained on photographs and
    paintings paired with hand-drawn line art (not on already-clean
    digital line art, as ``anime2sketch`` was), so it tends to suit
    photographed illustrations and paintings better -- see the README's
    "Conversion styles" section for a fuller comparison.
    """

    name = "informative_drawings"

    def __init__(
        self,
        weights_path: str | Path | None = None,
        n_residual_blocks: int = 3,
        detect_resolution: int = 1024,
        postprocess_strategy: Literal["nms", "hysteresis"] = "hysteresis",
    ) -> None:
        """Store where to find the pretrained weights and inference options.

        Parameters
        ----------
        weights_path : str | Path | None, optional
            Path to the ``.pth`` weights file. If None, resolved via
            :func:`_resolve_weights_path` (env var, then
            ``./weights/informative_drawings.pth``, then the per-user
            cache dir).
        n_residual_blocks : int, optional
            Number of residual blocks in the generator, by default 3,
            matching the upstream project's released checkpoints. Must
            match whatever checkpoint ``weights_path`` points to, or
            loading its ``state_dict`` will fail.
        detect_resolution : int, optional
            Target size, in pixels, for the image's *short* side before
            being fed through the network (see :func:`_resize_short_side`),
            by default 1024. Previously this resized to a 256x256
            *square*, which both distorted geometry (squashing every
            aspect ratio to 1:1) and produced a soft output too coarse to
            skeletonize cleanly (see ``postprocess_strategy`` below and
            ``docs/DIAGNOSIS.md`` #3). Named to match
            ``controlnet_aux.LineartDetector``'s own parameter, whose
            preprocessing this mirrors -- not ``load_size``, since it no
            longer targets a square.
        postprocess_strategy : {"nms", "hysteresis"}, optional
            Centerline-extraction strategy passed to
            :func:`~coloring_page.postprocess.soft_map_to_line_art`, by
            default ``"hysteresis"``, matching that function's own
            default. This is a forced override, not a re-validated
            choice: measured with ``scripts/compare.py`` at
            ``detect_resolution=1024``, ``"nms"``
            (:func:`~coloring_page.postprocess.gradient_edges`) still ties
            or wins on ``f1_normalized`` on all 3 reference pairs (0.500
            vs 0.503, 0.678 vs 0.609, 0.265 vs 0.230) -- its higher recall
            still outweighs its lower precision here, because doubling
            every stroke into its two edges (see ``gradient_edges``'s
            docstring) inflates ink coverage without leaving the
            admissibility band. The default is set to ``"hysteresis"``
            anyway, for consistency with ``soft_map_to_line_art`` and
            because a doubled-edge output is not a centerline regardless
            of what it scores. Pass ``postprocess_strategy="nms"``
            explicitly to restore the higher-scoring behavior. Revisit if
            ``hysteresis_centerline``'s thresholds are retuned for this
            network's output, or a sharper soft map becomes available.
        """
        self.weights_path = _resolve_weights_path(weights_path)
        self.n_residual_blocks = n_residual_blocks
        self.detect_resolution = detect_resolution
        self.postprocess_strategy = postprocess_strategy
        self._model: Generator | None = None

    def _get_model(self) -> Generator:
        """Lazily build the network and load its pretrained weights.

        Raises
        ------
        FileNotFoundError
            If no weights file exists at ``self.weights_path``, with
            instructions for obtaining one.
        RuntimeError
            If the checkpoint at ``self.weights_path`` doesn't match the
            generator architecture (e.g. wrong ``n_residual_blocks``),
            naming the specific missing/unexpected keys.
        """
        if self._model is not None:
            return self._model

        if not self.weights_path.exists():
            raise FileNotFoundError(_WEIGHTS_HELP.format(path=self.weights_path))

        model = build_generator(self.n_residual_blocks)
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        # strict=False so a mismatch reports exactly which keys are
        # missing/unexpected instead of just failing -- the default
        # strict=True path swallows that detail into a generic message.
        incompatible = model.load_state_dict(checkpoint, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint at {self.weights_path} does not match the "
                f"generator architecture (n_residual_blocks={self.n_residual_blocks}).\n"
                f"missing_keys: {incompatible.missing_keys}\n"
                f"unexpected_keys: {incompatible.unexpected_keys}"
            )
        model.eval()

        self._model = model
        return model

    def _infer(self, image: np.ndarray) -> np.ndarray:
        """Run the network at ``detect_resolution``, without resizing the result.

        Shared by :meth:`soft_map` (which upscales the result back to
        ``image``'s own size, for callers like
        :class:`~coloring_page.engines.gated.GatedEngine` that need it
        pixel-aligned with other same-resolution masks) and :meth:`convert`
        (which keeps it at this sharper working resolution through
        binarization, only resizing the final redrawn line art -- see
        that method's docstring for why the difference matters).

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Single-channel ``uint8`` image at the network's own working
            resolution (short side ``detect_resolution``, rounded to a
            multiple of 64 -- see :func:`_resize_short_side`), lower
            values = more ink-like (project convention).
        """
        model = self._get_model()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = _resize_short_side(rgb, self.detect_resolution)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = model(tensor)

        # The generator's Sigmoid output is already in [0, 1] with the
        # same polarity this project uses (0 = ink, 1 = background), the
        # same convention the upstream project itself relies on when
        # saving output directly via torchvision's save_image -- no
        # inversion needed, just rescaling to uint8.
        sketch = output.squeeze().clamp(0, 1).cpu().numpy()
        return cast(np.ndarray, (sketch * 255).astype(np.uint8))

    def soft_map(self, image: np.ndarray) -> np.ndarray:
        """Run the pretrained network and return its raw soft grayscale output.

        Exposed separately from :meth:`convert` so other code (e.g.
        :class:`~coloring_page.engines.gated.GatedEngine`, which uses this
        network's confidence as a semantic filter rather than a
        line-art source in its own right) can get the raw signal without
        going through :func:`~coloring_page.postprocess.soft_map_to_line_art`.

        Parameters
        ----------
        image : np.ndarray
            BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            Single-channel ``uint8`` image, same ``(H, W)`` as ``image``,
            lower values = more ink-like (project convention).
        """
        original_height, original_width = image.shape[:2]
        sketch = self._infer(image)
        upscaled: np.ndarray = cv2.resize(
            sketch, (original_width, original_height), interpolation=cv2.INTER_LINEAR
        )
        return upscaled

    def convert(
        self, image: np.ndarray, *, line_thickness: int = 1, debug: DebugSink | None = None
    ) -> np.ndarray:
        """Run the pretrained network and render its output as line art.

        Binarizes and redraws the network's soft output at its own
        working resolution, then resizes the *finished* line art back to
        ``image``'s size -- not the other way around. Upscaling the raw
        soft map first (this engine's previous behavior) turns each
        network-resolution stroke into a several-pixel-wide blur before
        it's ever thresholded, which is what made a clean centerline
        extraction (:func:`~coloring_page.postprocess.hysteresis_centerline`)
        impossible; resizing the already-thin, already-redrawn geometry
        instead preserves it.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        sketch = self._infer(image)
        if debug is not None:
            debug.save("raw_sketch", sketch)

        line_art = soft_map_to_line_art(
            sketch, strategy=self.postprocess_strategy, line_thickness=line_thickness
        )
        original_height, original_width = image.shape[:2]
        if line_art.shape[:2] != (original_height, original_width):
            line_art = cv2.resize(
                line_art, (original_width, original_height), interpolation=cv2.INTER_LINEAR
            )
        return line_art
