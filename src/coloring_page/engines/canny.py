"""Canny edge-detection conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine
from coloring_page.pipeline import remove_short_strokes, smooth_preserving_edges


class CannyEngine(ConversionEngine):
    """Produce line art with OpenCV's Canny edge detector.

    Canny works well on photos with clear, high-contrast boundaries
    (objects, faces, buildings) and tends to give crisp, thin outlines
    with relatively little noise, at the cost of sometimes missing soft
    or low-contrast edges that ``adaptive`` or ``xdog`` would pick up.
    """

    name = "canny"

    def __init__(self, low_threshold: int | None = None, high_threshold: int | None = None) -> None:
        """Store the Canny hysteresis thresholds used on every conversion.

        Parameters
        ----------
        low_threshold : int | None, optional
            Lower hysteresis threshold passed to ``cv2.Canny``, by default
            None. When None, it is computed per-image from the denoised
            image's median intensity (the "auto Canny" heuristic), which
            adapts edge sensitivity to each photo's actual contrast
            instead of using one fixed value for every image.
        high_threshold : int | None, optional
            Upper hysteresis threshold passed to ``cv2.Canny``, by default
            None, auto-computed the same way as ``low_threshold``.
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def convert(self, image: np.ndarray, *, line_thickness: int = 1) -> np.ndarray:
        """Detect edges with Canny and render them as black lines on white.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = smooth_preserving_edges(gray)

        low, high = self.low_threshold, self.high_threshold
        if low is None or high is None:
            # Auto-Canny: pick thresholds relative to the image's own
            # median intensity instead of a fixed pair that only suits
            # some photos' contrast.
            median = float(np.median(denoised))
            sigma = 0.33
            low = int(max(0, (1.0 - sigma) * median))
            high = int(min(255, (1.0 + sigma) * median))

        edges = cv2.Canny(denoised, low, high)

        # Bridge small hairline breaks left by the edge detector before
        # they get interpreted as separate, disconnected strokes.
        close_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        # Canny returns white edges on black; coloring pages need the
        # opposite (black lines on a white, printable background).
        lines = cv2.bitwise_not(edges)
        return remove_short_strokes(lines)
