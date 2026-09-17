"""Region-segmentation conversion engine for painterly/illustrated sources."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine
from coloring_page.pipeline import remove_small_specks


class CartoonEngine(ConversionEngine):
    """Produce line art by drawing outlines around flat, segmented regions.

    ``canny``, ``adaptive``, and ``xdog`` all key off local intensity
    *gradients*, which works well on photos but produces heavy stippling
    noise on painterly or textured illustrations (brush strokes, soft
    shading, glow effects all have internal gradients even though they're
    not object boundaries). This engine instead segments the image into
    flat regions with mean-shift filtering, then draws a line only where
    two *different* regions meet -- so internal texture, having been
    merged into a flat region, produces no line at all.

    Mean-shift filtering (``cv2.pyrMeanShiftFiltering``) merges pixels
    using *both* color similarity and spatial proximity, iteratively,
    which keeps smoothly-varying regions (sky gradients, glow/lighting
    effects) as a handful of large, contiguous blobs. A simpler global
    color quantization (e.g. k-means) instead slices a smooth gradient
    into many concentric color bands, each with its own boundary --
    producing exactly the ring-shaped background noise this engine is
    meant to avoid.
    """

    name = "cartoon"

    def __init__(
        self, spatial_radius: int = 25, color_radius: int = 48, min_region_area: int = 6
    ) -> None:
        """Store the mean-shift segmentation and cleanup parameters.

        Parameters
        ----------
        spatial_radius : int, optional
            Spatial window radius (in pixels) used by mean-shift
            filtering, by default 25. Larger values merge regions across
            a wider area, giving simpler, bolder outlines at the cost of
            fine detail.
        color_radius : int, optional
            Color window radius used by mean-shift filtering, by default
            48. Larger values merge pixels that differ more in color,
            further reducing gradient/texture noise but also erasing
            subtler color boundaries.
        min_region_area : int, optional
            Minimum connected-component size (in pixels) for a boundary
            fragment to survive cleanup, by default 6. Passed through to
            :func:`coloring_page.pipeline.remove_small_specks`.
        """
        self.spatial_radius = spatial_radius
        self.color_radius = color_radius
        self.min_region_area = min_region_area

    def convert(self, image: np.ndarray, *, line_thickness: int = 1) -> np.ndarray:
        """Segment the image into flat regions and outline their boundaries.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        segmented = cv2.pyrMeanShiftFiltering(image, self.spatial_radius, self.color_radius)
        gray_segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

        # A region boundary is any pixel where the segmented (now flat)
        # neighborhood isn't uniform -- the morphological gradient of a
        # piecewise-constant image is exactly its region outlines, with
        # zero response from anything that got merged into one flat region.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(gray_segmented, cv2.MORPH_GRADIENT, kernel)
        _, edges = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)

        if line_thickness > 1:
            thick_kernel = np.ones((line_thickness, line_thickness), np.uint8)
            edges = cv2.dilate(edges, thick_kernel, iterations=1)

        lines = cv2.bitwise_not(edges)
        return remove_small_specks(lines, min_area=self.min_region_area)
