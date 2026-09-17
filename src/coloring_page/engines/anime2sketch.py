"""Optional pretrained-model conversion engine (Anime2Sketch).

Requires the ``ml`` extra (``pip install -e ".[ml]"``) for ``torch``, and a
manually downloaded pretrained weights file -- neither is required by, or
bundled with, the rest of this project. See the README's "Extending with
an ML engine" section for setup instructions and license information.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from coloring_page.engines._anime2sketch_arch import UnetGenerator, build_generator
from coloring_page.engines.base import ConversionEngine
from coloring_page.pipeline import remove_small_specks

#: Environment variable used to point at a local weights file, in place of
#: the default lookup locations below.
WEIGHTS_ENV_VAR = "COLORING_PAGE_ANIME2SKETCH_WEIGHTS"

#: Where weights are looked for when neither a constructor argument nor
#: WEIGHTS_ENV_VAR is set: a `weights/` folder relative to the current
#: working directory (matching the upstream Anime2Sketch project's own
#: convention) first, then a per-user cache directory.
DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "coloring_page" / "anime2sketch.pth"
_LOCAL_WEIGHTS_PATH = Path("weights") / "anime2sketch.pth"


def _resolve_weights_path(explicit_path: str | Path | None) -> Path:
    """Pick the weights file to use, in priority order.

    Priority: an explicit path (constructor argument) > the
    ``COLORING_PAGE_ANIME2SKETCH_WEIGHTS`` environment variable > an
    existing ``./weights/anime2sketch.pth`` relative to the current working
    directory > the per-user cache directory (used as the final fallback
    even if it doesn't exist yet, so callers get a consistent "expected"
    path to report in error messages).
    """
    if explicit_path is not None:
        return Path(explicit_path)

    env_path = os.environ.get(WEIGHTS_ENV_VAR)
    if env_path:
        return Path(env_path)

    if _LOCAL_WEIGHTS_PATH.exists():
        return _LOCAL_WEIGHTS_PATH

    return DEFAULT_WEIGHTS_PATH


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

    def __init__(
        self,
        weights_path: str | Path | None = None,
        load_size: int = 512,
        gamma: float = 1.6,
        binarize: bool = True,
    ) -> None:
        """Store where to find the pretrained weights and inference options.

        Parameters
        ----------
        weights_path : str | Path | None, optional
            Path to the ``.pth`` weights file. If None, resolved via
            :func:`_resolve_weights_path` (env var, then
            ``./weights/anime2sketch.pth``, then the per-user cache dir).
        load_size : int, optional
            Square size (in pixels) the image is resized to before being
            fed through the network, by default 512, matching the size the
            upstream pretrained weights were trained/tested with. Must be
            a multiple of 256 (the network has 8 downsampling stages).
        gamma : float, optional
            Gamma-correction factor applied to the input image before
            inference, by default 1.6. On source images with large dark
            or shadowed regions, this pretrained network otherwise tends
            to collapse those regions into a solid black blob instead of
            linework; lifting shadow detail beforehand (`output = input **
            (1/gamma)`) avoids that failure mode. Set to 1.0 to disable.
        binarize : bool, optional
            Whether to threshold the network's soft, pencil-shaded output
            into crisp black-on-white ink lines (Otsu's method, after a
            percentile contrast stretch), by default True. This matches a
            printed coloring-book page far better than the raw grayscale
            sketch; set to False to keep the soft shading instead.
        """
        self.weights_path = _resolve_weights_path(weights_path)
        self.load_size = load_size
        self.gamma = gamma
        self.binarize = binarize
        self._model: UnetGenerator | None = None

    def _get_model(self) -> UnetGenerator:
        """Lazily build the network and load its pretrained weights.

        Raises
        ------
        FileNotFoundError
            If no weights file exists at ``self.weights_path``, with
            instructions for obtaining one.
        """
        if self._model is not None:
            return self._model

        if not self.weights_path.exists():
            raise FileNotFoundError(_WEIGHTS_HELP.format(path=self.weights_path))

        model = build_generator()
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
        model.eval()

        self._model = model
        return model

    def convert(self, image: np.ndarray, *, line_thickness: int = 1) -> np.ndarray:
        """Run the pretrained network and render its output as line art.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        model = self._get_model()
        original_height, original_width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        if self.gamma != 1.0:
            resized = (255 * (resized / 255.0) ** (1.0 / self.gamma)).astype(np.uint8)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5  # [0, 1] -> [-1, 1]

        with torch.no_grad():
            output = model(tensor)

        # Network output is a single-channel Tanh activation in [-1, 1];
        # rescale to a standard [0, 255] grayscale line-art image.
        sketch = output.squeeze().clamp(-1, 1).cpu().numpy()
        sketch = ((sketch + 1) / 2 * 255).astype(np.uint8)
        sketch = cv2.resize(
            sketch, (original_width, original_height), interpolation=cv2.INTER_CUBIC
        )

        if self.binarize:
            low, high = np.percentile(sketch, (2, 98))
            stretched = np.clip(
                (sketch.astype(np.float32) - low) / max(high - low, 1) * 255, 0, 255
            )
            sketch = stretched.astype(np.uint8)
            _, sketch = cv2.threshold(sketch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            sketch = cv2.erode(sketch, kernel, iterations=1)

        return remove_small_specks(sketch)
