"""Shared soft-mask binarization for ML engines that output grayscale line art.

A pretrained network like ``anime2sketch`` or ``informative_drawings``
outputs a *soft* grayscale map, not a decision about where a stroke is. A
single global threshold on a soft map is worse than useless here -- a
10px-wide dark region (a real stroke the network rendered with some width
or falloff) becomes a 10px-wide solid block of ink instead of the single
line it should read as. This module's functions turn that soft map into a
clean 1px centerline mask (:func:`hysteresis_centerline` or
:func:`gradient_edges`), which the ML engines then trace into vector paths
with :func:`coloring_page.drawing.paths_from_mask`.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from coloring_page.drawing import NEIGHBOR_KERNEL as _NEIGHBOR_KERNEL
from coloring_page.drawing import NEIGHBOR_OFFSETS as _NEIGHBOR_OFFSETS


def normalize_percentile(
    gray: np.ndarray, *, low_pct: float = 2.0, high_pct: float = 98.0
) -> np.ndarray:
    """Contrast-stretch a soft grayscale map using percentile bounds.

    A pretrained network's raw output rarely spans the full ``[0, 255]``
    range, so a fixed threshold behaves inconsistently across images.
    Percentile bounds (rather than the min/max) avoid a handful of
    extreme outlier pixels compressing the useful range.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel grayscale image, shape ``(H, W)``, dtype ``uint8``.
    low_pct : float, optional
        Lower percentile mapped to ``0``, by default 2.0.
    high_pct : float, optional
        Upper percentile mapped to ``255``, by default 98.0.

    Returns
    -------
    np.ndarray
        Contrast-stretched copy of ``gray``, same shape and dtype.
    """
    low, high = (float(bound) for bound in np.percentile(gray, (low_pct, high_pct)))
    if high <= low:
        # No usable contrast (e.g. a flat/near-flat map): there's no
        # evidence of a stroke anywhere, so treat it as pure background
        # rather than defaulting the whole image to "ink" (dividing by
        # the clamped denominator would otherwise map every pixel to 0).
        return np.full_like(gray, 255)
    stretched = np.clip((gray.astype(np.float32) - low) / (high - low) * 255, 0, 255)
    return cast(np.ndarray, stretched.astype(np.uint8))


def hysteresis_threshold(
    gray: np.ndarray, *, strong_threshold: float, weak_threshold: float
) -> np.ndarray:
    """Binarize a soft grayscale sketch, keeping weak ink connected to strong ink.

    A single global Otsu threshold picks one cutoff for the whole image,
    which is a poor fit here: the sketch's histogram is dominated (often
    ~95%) by near-white background, so Otsu's variance-splitting
    criterion tends to land in a spot that either keeps faint pencil
    lines as isolated noise or discards them outright. Mirroring how
    ``cv2.Canny`` itself avoids this (two thresholds plus connectivity)
    keeps faint linework that is actually connected to strong linework,
    while still dropping faint, isolated specks.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel grayscale sketch, shape ``(H, W)``, dtype
        ``uint8``, where lower values are darker (more ink-like).
    strong_threshold : float
        Pixel intensity at or below which a pixel is unambiguous ink.
    weak_threshold : float
        Pixel intensity at or below which a pixel is a candidate for ink
        if connected to a strong pixel. Must be ``>= strong_threshold``.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, ``0`` for kept ink pixels and
        ``255`` for background, same shape as ``gray``.
    """
    strong_mask = gray <= strong_threshold
    weak_mask = gray <= weak_threshold

    num_labels, labels = cv2.connectedComponents(weak_mask.astype(np.uint8), connectivity=8)
    strong_labels = np.unique(labels[strong_mask])
    strong_labels = strong_labels[strong_labels != 0]

    keep = np.isin(labels, strong_labels)
    return np.where(keep, 0, 255).astype(np.uint8)


def prune_short_branches(skeleton: np.ndarray, *, min_branch_length: int) -> np.ndarray:
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


def hysteresis_centerline(
    gray: np.ndarray,
    *,
    strong_threshold: float | None = None,
    weak_threshold: float | None = None,
    min_branch_length: int = 15,
) -> np.ndarray:
    """Reduce a soft grayscale map to a pruned 1px-wide centerline mask.

    Thresholds ``gray`` with :func:`hysteresis_threshold`, thins the
    surviving ink to a 1px skeleton with ``cv2.ximgproc.thinning``, then
    prunes short dead-end spurs with :func:`prune_short_branches`. A wide
    soft blob (the failure mode raw thresholding produces -- see the
    module docstring) collapses to the single line through its middle
    instead of staying a solid block.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel grayscale image, shape ``(H, W)``, dtype
        ``uint8``, where lower values are darker (more ink-like).
        Normalize with :func:`normalize_percentile` first: the default
        thresholds below are absolute values on that normalized 0-255
        scale, not percentiles of ``gray`` itself -- a *percentile* of
        the raw map is unreliable here because ink is typically a small
        minority of pixels (target coverage ~4-8%), so a percentile
        cutoff like "15th percentile" can land inside the background
        instead of the ink whenever ink covers less than 15% of the
        image.
    strong_threshold : float | None, optional
        Passed to :func:`hysteresis_threshold`. If None (the default),
        80.0 is used (out of the normalized 0-255 scale).
    weak_threshold : float | None, optional
        Passed to :func:`hysteresis_threshold`. If None (the default),
        160.0 is used (out of the normalized 0-255 scale).
    min_branch_length : int, optional
        Passed to :func:`prune_short_branches`, by default 15.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, ``255`` for a surviving
        centerline pixel and ``0`` for background.
    """
    if strong_threshold is None:
        strong_threshold = 80.0
    if weak_threshold is None:
        weak_threshold = 160.0

    binary = hysteresis_threshold(
        gray, strong_threshold=strong_threshold, weak_threshold=weak_threshold
    )
    ink_mask = np.where(binary == 0, 255, 0).astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(ink_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return prune_short_branches(skeleton, min_branch_length=min_branch_length)


def gradient_edges(
    gray: np.ndarray, *, high_threshold_percentile: float = 90.0, low_threshold_ratio: float = 0.4
) -> np.ndarray:
    """Reduce a soft grayscale map to its gradient-magnitude edges via Canny-style NMS.

    Treats ``gray`` as if it were a photo and runs it through the same
    gradient-direction non-maximum suppression plus hysteresis threshold
    ``cv2.Canny`` uses internally (its stage 2-3). This finds *edges*, in
    the image-processing sense: locations where intensity changes sharply.

    That is not the same thing as a stroke's centerline, and using it as
    one is a bug this project shipped as its default for a while (see
    ``docs/DIAGNOSIS.md`` §2): a soft stroke rendered with any width or
    falloff has *two* edges, one on each side, so this function reports
    two parallel lines for every real stroke instead of the one line
    down its middle. Formerly named ``nms_centerline``, which is exactly
    the misleading name that made shipping it as the centerline-extraction
    default plausible in the first place. Kept available as a legitimate
    edge detector -- e.g. for content that genuinely is a gradient
    transition rather than a hand-drawn stroke -- but the ML engines
    default their own ``postprocess_strategy`` to ``"hysteresis"``; use
    :func:`hysteresis_centerline` to actually collapse a wide soft stroke
    to its centerline.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel grayscale image, shape ``(H, W)``, dtype
        ``uint8``, where lower values are darker (more ink-like).
        Normalize with :func:`normalize_percentile` first for consistent
        thresholds across images.
    high_threshold_percentile : float, optional
        Percentile of *gradient magnitude* (not raw intensity -- Canny's
        thresholds compare against gradient magnitude, so deriving them
        from anything else compares mismatched scales) used as Canny's
        high threshold, by default 90.0. Mirrors
        :class:`~coloring_page.engines.skeleton_redraw.SkeletonRedrawEngine`'s
        own auto-threshold approach.
    low_threshold_ratio : float, optional
        Canny low threshold as a fraction of the high threshold, by
        default 0.4.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, ``255`` for a surviving
        centerline pixel and ``0`` for background.
    """
    darkness = (255 - gray.astype(np.int16)).astype(np.uint8)
    gx = cv2.Sobel(darkness, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(darkness, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    high_threshold = float(np.percentile(magnitude, high_threshold_percentile))
    low_threshold = high_threshold * low_threshold_ratio
    return cv2.Canny(darkness, low_threshold, high_threshold, L2gradient=True)


def redraw_segments(
    segments: list[np.ndarray],
    shape: tuple[int, int],
    *,
    polyline_epsilon: float = 1.2,
    line_thickness: int = 1,
) -> np.ndarray:
    """Smooth and redraw already-traced point-chain segments at a uniform width.

    Takes input that is already a list of connected point sequences (e.g.
    ``cv2.ximgproc.EdgeDrawing``'s ``getSegments()``) rather than a binary
    mask needing tracing. Used for diagnostic raster previews (e.g.
    :class:`~coloring_page.engines.gated.GatedEngine`'s debug stages) --
    the final vector output of those engines instead goes through
    :func:`coloring_page.drawing.paths_from_point_chains`.

    Parameters
    ----------
    segments : list[np.ndarray]
        Point-chain segments, each an ``Nx1x2`` or ``Nx2`` int32 array.
    shape : tuple[int, int]
        ``(height, width)`` of the output canvas.
    polyline_epsilon : float, optional
        ``cv2.approxPolyDP`` tolerance used to smooth each segment before
        redrawing, in pixels, by default 1.2.
    line_thickness : int, optional
        Output stroke width in pixels, by default 1.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, project convention: ``0`` = ink,
        ``255`` = background.
    """
    canvas = np.full(shape, 255, dtype=np.uint8)
    for segment in segments:
        approx = cv2.approxPolyDP(segment, polyline_epsilon, closed=False)
        cv2.polylines(
            canvas,
            [approx],
            isClosed=False,
            color=0,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas
