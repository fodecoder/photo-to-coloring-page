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

import cv2
import numpy as np
import torch

from coloring_page.engines._informative_drawings_arch import Generator, build_generator
from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.pipeline import remove_short_strokes

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


_WEIGHTS_HELP = (
    "Informative Drawings pretrained weights not found at {path}.\n"
    "This engine requires weights that this project does not bundle or "
    "download automatically:\n"
    "  1. Download 'model.zip' from the official Google Drive link in the "
    "Informative Drawings README:\n"
    "     https://github.com/carolineec/informative-drawings#testing\n"
    "  2. Unzip it and locate checkpoints/contour_style/netG_A_latest.pth "
    "(or another style's netG_A_latest.pth -- contour_style is "
    "recommended for photographed/printed illustrations).\n"
    f"  3. Save it to {_LOCAL_WEIGHTS_PATH} (relative to the current "
    f"directory), {DEFAULT_WEIGHTS_PATH}, or set the {WEIGHTS_ENV_VAR} "
    "environment variable to wherever you saved it.\n"
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
        load_size: int = 256,
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
        load_size : int, optional
            Square size (in pixels) the image is resized to before being
            fed through the network, by default 256, matching the size
            the upstream pretrained checkpoints were trained/tested
            with.
        """
        self.weights_path = _resolve_weights_path(weights_path)
        self.n_residual_blocks = n_residual_blocks
        self.load_size = load_size
        self._model: Generator | None = None

    def _get_model(self) -> Generator:
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

        model = build_generator(self.n_residual_blocks)
        checkpoint = torch.load(self.weights_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        model.eval()

        self._model = model
        return model

    def convert(
        self, image: np.ndarray, *, line_thickness: int = 1, debug: DebugSink | None = None
    ) -> np.ndarray:
        """Run the pretrained network and render its output as line art.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        model = self._get_model()
        original_height, original_width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC
        )
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = model(tensor)

        # The generator's Sigmoid output is already in [0, 1] with the
        # same polarity this project uses (0 = ink, 1 = background), the
        # same convention the upstream project itself relies on when
        # saving output directly via torchvision's save_image -- no
        # inversion needed, just rescaling to uint8.
        sketch = output.squeeze().clamp(0, 1).cpu().numpy()
        sketch = (sketch * 255).astype(np.uint8)
        sketch = cv2.resize(
            sketch, (original_width, original_height), interpolation=cv2.INTER_CUBIC
        )
        if debug is not None:
            debug.save("raw_sketch", sketch)

        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            sketch = cv2.erode(sketch, kernel, iterations=1)

        return remove_short_strokes(sketch)
