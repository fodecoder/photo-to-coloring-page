"""Canny edge-detection conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine


class CannyEngine(ConversionEngine):
    """Produce line art with OpenCV's Canny edge detector.

    Canny works well on photos with clear, high-contrast boundaries
    (objects, faces, buildings) and tends to give crisp, thin outlines
    with relatively little noise, at the cost of sometimes missing soft
    or low-contrast edges that ``adaptive`` or ``xdog`` would pick up.
    """

    name = "canny"

    def __init__(self, low_threshold: int = 50, high_threshold: int = 150) -> None:
        """Store the Canny hysteresis thresholds used on every conversion.

        Parameters
        ----------
        low_threshold : int, optional
            Lower hysteresis threshold passed to ``cv2.Canny``, by default 50.
        high_threshold : int, optional
            Upper hysteresis threshold passed to ``cv2.Canny``, by default 150.
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
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)

        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        # Canny returns white edges on black; coloring pages need the
        # opposite (black lines on a white, printable background).
        return cv2.bitwise_not(edges)
