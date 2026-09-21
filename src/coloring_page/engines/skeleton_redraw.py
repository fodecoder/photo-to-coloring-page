"""Flatten-skeletonize-redraw conversion engine, an alternative to ``chained``."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.drawing import Drawing, paths_from_mask, rasterize
from coloring_page.engines.base import ConversionEngine, DebugSink


class SkeletonRedrawEngine(ConversionEngine):
    """Produce line art via flatten -> skeletonize -> redraw.

    An alternative to :class:`~coloring_page.engines.chained.ChainedEngine`
    that reaches a comparable "redraw the geometry, not the mask" result
    through a different route: instead of EdgeDrawing's chained edges,
    it thresholds an L0-smoothed edge map, thins it to a 1px-wide
    skeleton (``cv2.ximgproc.thinning``), prunes the short spurious
    branches thinning introduces at junctions, then traces and redraws
    each remaining skeleton contour. See ``scripts/compare.py`` for how
    to compare this against ``chained`` on real images before choosing
    a default.
    """

    name = "skeleton"
    experimental = True

    def __init__(
        self,
        l0_lambda: float = 0.02,
        l0_kappa: float = 2.0,
        gradient_percentile: float = 95.0,
        low_threshold_ratio: float = 0.4,
        min_branch_length: int = 15,
        polyline_epsilon: float = 1.2,
    ) -> None:
        """Store the flatten/skeletonize/redraw stage parameters.

        Parameters
        ----------
        l0_lambda : float, optional
            L0 gradient minimization's smoothness weight, by default
            0.02. Larger values flatten more aggressively.
        l0_kappa : float, optional
            L0 gradient minimization's weight-increase rate, by default
            2.0.
        gradient_percentile : float, optional
            Percentile of Sobel gradient magnitude used as the Canny
            high threshold, by default 95.0 (mirrors
            :class:`~coloring_page.engines.canny.CannyEngine`'s
            auto-threshold approach, but higher: a raw Canny edge map
            feeding a skeletonizer needs to start cleaner than one
            feeding EdgeDrawing's own chain-length filtering).
        low_threshold_ratio : float, optional
            Canny low threshold as a fraction of the high threshold, by
            default 0.4.
        min_branch_length : int, optional
            Minimum surviving branch length, in pixels, passed to
            :func:`prune_short_branches`, by default 15.
        polyline_epsilon : float, optional
            ``cv2.approxPolyDP`` tolerance used to smooth each traced
            contour before redrawing, in pixels, by default 1.2.
        """
        self.l0_lambda = l0_lambda
        self.l0_kappa = l0_kappa
        self.gradient_percentile = gradient_percentile
        self.low_threshold_ratio = low_threshold_ratio
        self.min_branch_length = min_branch_length
        self.polyline_epsilon = polyline_epsilon

    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> Drawing:
        """Flatten, then trace the edge map's skeleton into smooth vector paths.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        flattened = cv2.ximgproc.l0Smooth(image, lambda_=self.l0_lambda, kappa=self.l0_kappa)
        if debug is not None:
            debug.save("flattened", flattened)
        gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        high = float(np.percentile(magnitude, self.gradient_percentile))
        low = high * self.low_threshold_ratio
        edges = cv2.Canny(gray, low, high, L2gradient=True)
        if debug is not None:
            debug.save("edges", edges)

        # paths_from_mask thins and traces edges' skeleton graph directly
        # (branch-to-branch, never via cv2.findContours), which is exactly
        # what this engine used to hand-roll via its own
        # thinning/prune_short_branches/findContours block -- that block
        # is now redundant with paths_from_mask and has been removed.
        working_dimension = max(gray.shape[:2])
        paths = paths_from_mask(edges)
        drawing = Drawing(paths=paths, aspect_ratio=image.shape[1] / image.shape[0])
        drawing = drawing.simplify(self.polyline_epsilon / working_dimension)
        min_length_normalized = self.min_branch_length / working_dimension
        drawing = drawing.filter(lambda p: p.length >= min_length_normalized)

        if debug is not None:
            debug.save("redraw", rasterize(drawing, long_side_px=working_dimension))
        return drawing
