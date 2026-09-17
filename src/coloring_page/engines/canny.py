"""Canny edge-detection conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine
from coloring_page.pipeline import derive_kernel_size, remove_short_strokes, smooth_preserving_edges


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
            None. When None, it is computed per-image from a high
            percentile of the gradient magnitude (see :meth:`convert`),
            which adapts edge sensitivity to each photo's actual edge
            strength instead of using one fixed value for every image.
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

        low: float | None
        high: float | None
        low, high = self.low_threshold, self.high_threshold
        if low is None or high is None:
            # Auto-Canny on median *intensity* assumes a centered
            # histogram; a light book-page photo has a median near white,
            # which pushes both thresholds so high that only the
            # strongest edges survive and most contours end up broken.
            # Thresholding on the gradient *magnitude*'s own distribution
            # instead adapts to how strong this specific photo's edges
            # actually are, regardless of its overall brightness.
            gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
            high = float(np.percentile(magnitude, 92.5))
            low = high * 0.4

        # L2gradient uses the true Euclidean gradient norm; the default
        # L1 approximation overstates diagonal edges relative to
        # horizontal/vertical ones.
        edges = cv2.Canny(denoised, low, high, L2gradient=True)

        # Bridge small hairline breaks left by the edge detector before
        # they get interpreted as separate, disconnected strokes.
        working_dimension = max(gray.shape[:2])
        close_size = derive_kernel_size(working_dimension, fraction=0.002, min_value=3)
        close_kernel = np.ones((close_size, close_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        # Canny returns white edges on black; coloring pages need the
        # opposite (black lines on a white, printable background).
        lines = cv2.bitwise_not(edges)
        min_extent = derive_kernel_size(working_dimension, fraction=0.012, min_value=3, odd=False)
        return remove_short_strokes(lines, min_extent=min_extent)
