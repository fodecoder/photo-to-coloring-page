"""Color-quantization conversion engine for painterly/illustrated sources."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine
from coloring_page.pipeline import remove_small_specks

# Iteration/precision settings for cv2.kmeans; the exact numbers matter far
# less than having *some* cap so quantization is fast and deterministic.
_KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)


class CartoonEngine(ConversionEngine):
    """Produce line art by drawing outlines around flat, quantized colors.

    ``canny``, ``adaptive``, and ``xdog`` all key off local intensity
    *gradients*, which works well on photos but produces heavy stippling
    noise on painterly or textured illustrations (brush strokes, soft
    shading, glow effects all have internal gradients even though they're
    not object boundaries). This engine instead quantizes the image down
    to a small palette of flat colors with k-means clustering, then draws
    a line only where two *different* quantized regions meet -- so
    internal texture, having no color change, produces no line at all.
    This is the standard "cartoonizer" technique and is a better match
    for illustrated source material than the other three engines.
    """

    name = "cartoon"

    def __init__(self, num_colors: int = 6, bilateral_passes: int = 3) -> None:
        """Store the palette size and pre-smoothing strength.

        Parameters
        ----------
        num_colors : int, optional
            Number of color clusters k-means quantizes the image to, by
            default 6. Fewer colors give simpler, bolder regions and
            fewer boundary lines; more colors preserve finer detail at
            the cost of extra lines and, on heavily-shaded source images,
            spurious lines splitting a single object into two colors.
        bilateral_passes : int, optional
            Number of edge-preserving bilateral-filter passes applied
            before quantization, by default 3. Repeated passes erase more
            paint/texture noise than one pass with an equivalently large
            kernel, while still keeping strong color boundaries sharp.
        """
        self.num_colors = num_colors
        self.bilateral_passes = bilateral_passes

    def _quantize_colors(self, image: np.ndarray) -> np.ndarray:
        """Reduce ``image`` to ``self.num_colors`` flat BGR colors via k-means."""
        samples = image.reshape((-1, 3)).astype(np.float32)
        _, labels, centers = cv2.kmeans(
            samples,
            self.num_colors,
            None,  # type: ignore[call-overload]
            _KMEANS_CRITERIA,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        quantized = centers[labels.flatten()].astype(np.uint8)
        return quantized.reshape(image.shape)  # type: ignore[no-any-return]

    def convert(self, image: np.ndarray, *, line_thickness: int = 1) -> np.ndarray:
        """Quantize colors and draw lines at the boundaries between regions.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        smoothed = image
        for _ in range(self.bilateral_passes):
            smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=60, sigmaSpace=60)

        quantized = self._quantize_colors(smoothed)
        gray_quantized = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)

        # A region boundary is any pixel where the quantized (now flat)
        # neighborhood isn't uniform -- the morphological gradient of a
        # piecewise-constant image is exactly its region outlines, with
        # zero response from anything that got quantized to one flat color.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(gray_quantized, cv2.MORPH_GRADIENT, kernel)
        _, edges = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)

        if line_thickness > 1:
            thick_kernel = np.ones((line_thickness, line_thickness), np.uint8)
            edges = cv2.dilate(edges, thick_kernel, iterations=1)

        lines = cv2.bitwise_not(edges)
        return remove_small_specks(lines)
