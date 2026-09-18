"""Shared soft-mask binarization for ML engines that output grayscale line art.

A pretrained network like ``anime2sketch`` or ``informative_drawings``
outputs a *soft* grayscale map, not a decision about where a stroke is.
Turning that into a coloring page needs the same care classical engines
already take: redrawing traced geometry at a uniform width, not thresholding
a mask and keeping its ragged edges. A single global threshold on a soft
map is worse than useless here -- a 10px-wide dark region (a real stroke
the network rendered with some width or falloff) becomes a 10px-wide solid
block of ink instead of the single line it should read as. This module
extracts that "soft map -> clean line art" pipeline so every ML engine
shares it rather than reimplementing its own ad hoc thresholding.
"""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np

from coloring_page.pipeline import remove_short_strokes

#: 8-neighborhood sum kernel (excludes the center pixel itself), used to
#: classify skeleton pixels as endpoints/junctions by neighbor count.
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

#: Offsets, in (dy, dx) order, to each of a pixel's 8 neighbors.
_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]  # fmt: skip


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


def nms_centerline(
    gray: np.ndarray, *, high_threshold_percentile: float = 90.0, low_threshold_ratio: float = 0.4
) -> np.ndarray:
    """Reduce a soft grayscale map to a centerline mask via Canny-style NMS.

    Treats ``gray`` as if it were a photo and runs it through the same
    gradient-direction non-maximum suppression plus hysteresis threshold
    ``cv2.Canny`` uses internally (its stage 2-3): a candidate edge pixel
    survives only if it's a local maximum of gradient magnitude measured
    *across* the local edge direction, which thins a wide soft transition
    down to a single pixel row without an explicit skeletonization pass.

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


def redraw_centerline(
    centerline: np.ndarray, *, polyline_epsilon: float = 1.2, line_thickness: int = 1
) -> np.ndarray:
    """Trace a thinned centerline mask and redraw it as smooth constant-width strokes.

    Mirrors ``chained.py``'s/``skeleton_redraw.py``'s redraw stage:
    extracts the *geometry* of each connected stroke with
    ``cv2.findContours``, smooths it with ``cv2.approxPolyDP``, and
    redraws it with ``cv2.polylines`` at a uniform width. Redrawing the
    geometry, not the raw mask, is what gives a uniform stroke width
    instead of the ragged edges a soft-mask threshold leaves behind.

    Parameters
    ----------
    centerline : np.ndarray
        Single-channel ``uint8`` image, ``255`` = line pixel, ``0`` =
        background, as produced by :func:`hysteresis_centerline` or
        :func:`nms_centerline`.
    polyline_epsilon : float, optional
        ``cv2.approxPolyDP`` tolerance used to smooth each traced
        contour before redrawing, in pixels, by default 1.2.
    line_thickness : int, optional
        Output stroke width in pixels, by default 1.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, project convention: ``0`` = ink,
        ``255`` = background.
    """
    contours, _ = cv2.findContours(centerline, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.full(centerline.shape, 255, dtype=np.uint8)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, polyline_epsilon, closed=False)
        cv2.polylines(
            canvas,
            [approx],
            isClosed=False,
            color=0,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas


def redraw_segments(
    segments: list[np.ndarray],
    shape: tuple[int, int],
    *,
    polyline_epsilon: float = 1.2,
    line_thickness: int = 1,
) -> np.ndarray:
    """Smooth and redraw already-traced point-chain segments at a uniform width.

    Like :func:`redraw_centerline`, but for input that is already a list
    of connected point sequences (e.g. ``cv2.ximgproc.EdgeDrawing``'s
    ``getSegments()``) rather than a binary mask -- skips the
    ``cv2.findContours`` tracing step :func:`redraw_centerline` needs,
    since these segments are already traced. Shared by
    :class:`~coloring_page.engines.chained.ChainedEngine` and
    :class:`~coloring_page.engines.gated.GatedEngine`.

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


def soft_map_to_line_art(
    soft: np.ndarray,
    *,
    strategy: Literal["nms", "hysteresis"] = "nms",
    line_thickness: int = 1,
    min_branch_length: int = 15,
    polyline_epsilon: float = 1.2,
) -> np.ndarray:
    """Turn a pretrained network's raw soft output into clean, printable line art.

    Top-level entry point combining this module's three stages:
    :func:`normalize_percentile`, a centerline-extraction strategy
    (:func:`hysteresis_centerline` or :func:`nms_centerline`), and
    :func:`redraw_centerline`. This is what every ML engine's
    ``convert()`` should call once it has a soft grayscale map, instead
    of thresholding and keeping the raw mask.

    Parameters
    ----------
    soft : np.ndarray
        Single-channel grayscale image, shape ``(H, W)``, dtype
        ``uint8``, lower = more ink-like -- a pretrained network's raw
        output, resized to the original image's dimensions.
    strategy : {"nms", "hysteresis"}, optional
        Which centerline-extraction strategy to use, by default
        ``"nms"``. Measured against this project's reference images,
        ``"nms"`` consistently recovers more true-positive ink than
        ``"hysteresis"`` (whose explicit skeletonize+prune step loses
        more real strokes than it removes noise) -- see
        ``scripts/compare.py`` before changing an engine's default.
    line_thickness : int, optional
        Output stroke width in pixels, by default 1.
    min_branch_length : int, optional
        Passed to :func:`hysteresis_centerline` (ignored by ``"nms"``),
        by default 15.
    polyline_epsilon : float, optional
        Passed to :func:`redraw_centerline`, by default 1.2.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, project convention: ``0`` = ink,
        ``255`` = background, with short-stroke noise removed.
    """
    normalized = normalize_percentile(soft)

    if strategy == "hysteresis":
        centerline = hysteresis_centerline(normalized, min_branch_length=min_branch_length)
    elif strategy == "nms":
        centerline = nms_centerline(normalized)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; expected 'nms' or 'hysteresis'.")

    line_art = redraw_centerline(
        centerline, polyline_epsilon=polyline_epsilon, line_thickness=line_thickness
    )
    return remove_short_strokes(line_art)
