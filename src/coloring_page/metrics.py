"""Quantitative quality metrics for coloring-page line-art output.

Tuning any of the conversion engines by eye doesn't scale and doesn't
catch regressions -- the XDoG and Canny bugs fixed in this project both
produced output that "looked plausible" as a diff but was quantifiably
wrong (0.05% ink coverage is not a coloring page). These metrics give a
numeric way to compare an engine's output against the project's target
range and to compare candidate engines against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def ink_coverage(binary_image: np.ndarray) -> float:
    """Fraction of pixels that are ink rather than background.

    Parameters
    ----------
    binary_image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background (paper) and darker pixels are ink.

    Returns
    -------
    float
        Ink pixel fraction in ``[0, 1]``. The project's target range for
        a finished coloring page is roughly 5-8%.
    """
    return float(np.mean(binary_image < 128))


def component_count(binary_image: np.ndarray, *, connectivity: int = 8) -> int:
    """Number of distinct connected ink components.

    Parameters
    ----------
    binary_image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background (paper) and darker pixels are ink.
    connectivity : int, optional
        Pixel connectivity used to group ink pixels into components, by
        default 8.

    Returns
    -------
    int
        Number of ink components (excluding the background). The
        project's target range for a finished coloring page is roughly
        100-200.
    """
    ink_mask = (binary_image < 128).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(ink_mask, connectivity=connectivity)
    return num_labels - 1


def median_stroke_length(binary_image: np.ndarray, *, connectivity: int = 8) -> float:
    """Median bounding-box extent across all ink components.

    Parameters
    ----------
    binary_image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background (paper) and darker pixels are ink.
    connectivity : int, optional
        Pixel connectivity used to group ink pixels into components, by
        default 8.

    Returns
    -------
    float
        Median of each component's bounding-box extent (the larger of
        its width/height), in pixels. ``0.0`` if there is no ink.
    """
    ink_mask = (binary_image < 128).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=connectivity)
    if len(stats) <= 1:
        return 0.0
    extents = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
    return float(np.median(extents))


def noise_fraction(
    binary_image: np.ndarray, *, min_extent: int = 4, connectivity: int = 8
) -> float:
    """Fraction of ink pixels belonging to components below ``min_extent``.

    A residual-noise indicator: even when overall ink coverage looks
    reasonable, a result can still be dominated by many small speckled
    fragments rather than a few coherent strokes. This measures how much
    of the ink is actually made up of such fragments.

    Parameters
    ----------
    binary_image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background (paper) and darker pixels are ink.
    min_extent : int, optional
        Bounding-box extent below which a component counts as noise, by
        default 4 (matching :func:`coloring_page.pipeline.remove_short_strokes`'s
        own default).
    connectivity : int, optional
        Pixel connectivity used to group ink pixels into components, by
        default 8.

    Returns
    -------
    float
        Fraction, in ``[0, 1]``, of total ink pixels that belong to
        components with an extent below ``min_extent``. ``0.0`` if there
        is no ink.
    """
    ink_mask = (binary_image < 128).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink_mask, connectivity=connectivity
    )
    total_ink = int(np.sum(ink_mask))
    if total_ink == 0:
        return 0.0

    extents = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    is_noise_label = extents < min_extent
    is_noise_label[0] = False  # background label is never "noise"

    noise_pixels = int(np.sum(is_noise_label[labels] & ink_mask.astype(bool)))
    return noise_pixels / total_ink


def boundary_f_measure(
    pred: np.ndarray, gt: np.ndarray, *, tolerance: int = 2
) -> tuple[float, float, float]:
    """Precision/recall/F1 of predicted strokes against a reference drawing.

    Pixel-wise equality is meaningless for line art: two drawings of the
    same scene never place strokes on identical pixels. The standard fix
    (the BSDS500 boundary benchmark) is to count a predicted ink pixel as
    correct when a reference ink pixel lies within ``tolerance``, and vice
    versa for recall. A distance transform gives this without the full
    bipartite matching of the original benchmark.

    ``pred`` and ``gt`` must already be at the same scale: ``tolerance`` is
    a pixel count, so it only means the same thing on both images if they
    share resolution. Use :func:`compare_boundaries` when the two inputs
    come from images of different sizes.

    Parameters
    ----------
    pred : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background and darker pixels are ink.
    gt : np.ndarray
        Reference image, same shape convention as ``pred`` and the same
        ``(H, W)`` shape.
    tolerance : int, optional
        Matching radius in pixels, by default 2. Wider tolerances reward
        drawing more ink rather than drawing the right ink -- see the
        degenerate-baseline tests in ``tests/test_metrics.py``, which is
        why the project default stays tight.

    Returns
    -------
    tuple[float, float, float]
        ``(precision, recall, f1)``, each in ``[0, 1]``.
    """
    pred_ink = (pred < 160).astype(np.uint8)
    gt_ink = (gt < 160).astype(np.uint8)

    dist_to_gt = cv2.distanceTransform(1 - gt_ink, cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform(1 - pred_ink, cv2.DIST_L2, 5)

    precision = float((dist_to_gt[pred_ink > 0] <= tolerance).mean()) if pred_ink.any() else 0.0
    recall = float((dist_to_pred[gt_ink > 0] <= tolerance).mean()) if gt_ink.any() else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compare_boundaries(
    pred: np.ndarray, gt: np.ndarray, *, long_side: int = 864, tolerance: int = 2
) -> tuple[float, float, float]:
    """Boundary precision/recall/F1 between two images of possibly different sizes.

    Resizes ``pred`` so its long side is ``long_side`` (preserving its own
    aspect ratio), then resizes ``gt`` to that exact resulting shape,
    before delegating to :func:`boundary_f_measure`. Normalizing scale
    first matters because ``tolerance`` is a pixel count: comparing a
    4000px photo against an 800px reference at the same nominal tolerance
    would be far stricter on the larger image. ``pred`` and ``gt`` must
    end up the *same* shape for a pixel-wise distance-transform match, so
    when the two images' aspect ratios differ (a near-crop reference pair
    rather than an exact one), ``gt`` is stretched slightly to fit --
    callers comparing mismatched-aspect pairs should treat the resulting
    score as approximate (see ``scripts/compare.py``'s aspect-ratio
    mismatch warning).

    Parameters
    ----------
    pred : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``.
    gt : np.ndarray
        Reference image, same single-channel convention as ``pred``.
    long_side : int, optional
        Target size, in pixels, for ``pred``'s longer dimension after
        resizing, by default 864 (the reference images' own long side, so
        an exact-aspect pair needs no resizing at all).
    tolerance : int, optional
        Matching radius in pixels at the normalized scale, by default 2.

    Returns
    -------
    tuple[float, float, float]
        ``(precision, recall, f1)``, each in ``[0, 1]``.
    """
    pred_height, pred_width = pred.shape[:2]
    scale = long_side / max(pred_height, pred_width)
    target_size = (max(1, round(pred_width * scale)), max(1, round(pred_height * scale)))

    resized_pred = cv2.resize(pred, target_size, interpolation=cv2.INTER_AREA)
    resized_gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_AREA)

    return boundary_f_measure(resized_pred, resized_gt, tolerance=tolerance)


@dataclass
class LineArtMetrics:
    """Bundle of the quality metrics computed for one line-art output."""

    ink_coverage: float
    component_count: int
    median_stroke_length: float
    noise_fraction: float


def compute_metrics(binary_image: np.ndarray, *, min_extent: int = 4) -> LineArtMetrics:
    """Compute all line-art quality metrics for a single output image.

    Parameters
    ----------
    binary_image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background (paper) and darker pixels are ink.
    min_extent : int, optional
        Extent threshold passed through to :func:`noise_fraction`, by
        default 4.

    Returns
    -------
    LineArtMetrics
        All four metrics for ``binary_image``.
    """
    return LineArtMetrics(
        ink_coverage=ink_coverage(binary_image),
        component_count=component_count(binary_image),
        median_stroke_length=median_stroke_length(binary_image),
        noise_fraction=noise_fraction(binary_image, min_extent=min_extent),
    )
