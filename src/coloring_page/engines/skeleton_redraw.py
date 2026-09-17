"""Flatten-skeletonize-redraw conversion engine, an alternative to ``chained``."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.pipeline import derive_kernel_size, remove_short_strokes

#: 8-neighborhood sum kernel (excludes the center pixel itself).
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

#: Offsets, in (dy, dx) order, to each of a pixel's 8 neighbors.
_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]  # fmt: skip


def _prune_short_branches(skeleton: np.ndarray, *, min_branch_length: int) -> np.ndarray:
    """Erase dead-end skeleton spurs shorter than ``min_branch_length``.

    Zhang-Suen thinning produces a spurious short dead-end branch at
    nearly every real junction/bifurcation in the source shape, which
    reads as a "hairy" or jittery stroke once redrawn. This walks the
    skeleton from each endpoint (a pixel with exactly one neighbor)
    toward the nearest junction (a pixel with three or more neighbors);
    if that walk is shorter than ``min_branch_length``, the walked
    pixels are erased, leaving the junction and the rest of the
    skeleton untouched.

    Parameters
    ----------
    skeleton : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` marks a skeleton pixel and ``0`` is background (as
        produced by ``cv2.ximgproc.thinning``).
    min_branch_length : int
        Branches with fewer pixels than this are erased.

    Returns
    -------
    np.ndarray
        Copy of ``skeleton`` with short dead-end branches erased.
    """
    mask = (skeleton > 0).astype(np.uint8)
    if not mask.any():
        return skeleton.copy()

    counts = cv2.filter2D(mask, -1, _NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT) * mask
    height, width = mask.shape
    result = mask.copy()

    endpoints = np.argwhere((mask == 1) & (counts == 1))
    for endpoint_y, endpoint_x in endpoints:
        if result[endpoint_y, endpoint_x] == 0:
            continue  # already erased by pruning an earlier, overlapping endpoint

        path = [(int(endpoint_y), int(endpoint_x))]
        visited = {(int(endpoint_y), int(endpoint_x))}
        reached_junction = False
        y, x = int(endpoint_y), int(endpoint_x)

        while len(path) <= min_branch_length:
            candidates = [
                (y + dy, x + dx)
                for dy, dx in _NEIGHBOR_OFFSETS
                if 0 <= y + dy < height
                and 0 <= x + dx < width
                and result[y + dy, x + dx]
                and (y + dy, x + dx) not in visited
            ]
            if len(candidates) != 1:
                break
            y, x = candidates[0]
            if counts[y, x] >= 3:
                reached_junction = True
                break
            path.append((y, x))
            visited.add((y, x))

        if reached_junction and len(path) < min_branch_length:
            for py, px in path:
                result[py, px] = 0

    return (result * 255).astype(skeleton.dtype)


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
            :func:`_prune_short_branches`, by default 15.
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

    def convert(
        self, image: np.ndarray, *, line_thickness: int = 1, debug: DebugSink | None = None
    ) -> np.ndarray:
        """Flatten, skeletonize, prune, then redraw as smooth strokes.

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

        skeleton = cv2.ximgproc.thinning(edges, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        pruned = _prune_short_branches(skeleton, min_branch_length=self.min_branch_length)
        if debug is not None:
            debug.save("skeleton", pruned)

        contours, _ = cv2.findContours(pruned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        canvas = np.full(gray.shape, 255, dtype=np.uint8)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, self.polyline_epsilon, closed=False)
            cv2.polylines(
                canvas, [approx], isClosed=False, color=0, thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )
        if debug is not None:
            debug.save("redraw", canvas)

        working_dimension = max(gray.shape[:2])
        min_extent = derive_kernel_size(working_dimension, fraction=0.018, min_value=3, odd=False)
        return remove_short_strokes(canvas, min_extent=min_extent)
