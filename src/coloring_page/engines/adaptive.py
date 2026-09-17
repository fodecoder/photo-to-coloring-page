"""Adaptive-thresholding conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine
from coloring_page.pipeline import remove_short_strokes, smooth_preserving_edges


class AdaptiveEngine(ConversionEngine):
    """Produce line art with a blurred image and adaptive thresholding.

    This mimics a classic "pencil sketch" technique: the image is
    heavily blurred, then thresholded per-neighborhood so that local
    contrast (rather than a single global cutoff) decides where a line
    falls. It tends to preserve more shading detail as texture than
    ``canny``, which can be desirable for portraits or busy scenes.
    """

    name = "adaptive"

    def __init__(self, block_size: int = 9, c: int = 2) -> None:
        """Store the adaptive-threshold neighborhood size and constant.

        Parameters
        ----------
        block_size : int, optional
            Size (in pixels) of the pixel neighborhood used to compute each
            threshold, by default 9. Must be odd; even values are bumped up
            by one.
        c : int, optional
            Constant subtracted from the computed mean/weighted mean, by
            default 2. Higher values produce thinner, sparser lines.
        """
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c = c

    def convert(self, image: np.ndarray, *, line_thickness: int = 1) -> np.ndarray:
        """Blur then adaptively threshold the image into line art.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = smooth_preserving_edges(gray)
        lines = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c,
        )
        lines = remove_short_strokes(lines)

        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            # Lines are dark on a light background here, so eroding
            # (rather than dilating) grows the dark strokes.
            lines = cv2.erode(lines, kernel, iterations=1)

        return lines
