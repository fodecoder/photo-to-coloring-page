"""Tests for the line-art quality metrics module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from coloring_page.engines.canny import CannyEngine
from coloring_page.engines.chained import ChainedEngine
from coloring_page.metrics import (
    boundary_f_measure,
    compare_boundaries,
    component_count,
    compute_metrics,
    degenerate_floor,
    ink_coverage,
    median_stroke_length,
    noise_fraction,
)

from .conftest import make_synthetic_photo


def _make_image() -> np.ndarray:
    """A 20x20 canvas with three known components: two 6x6 blocks and a 2px dot."""
    image = np.full((20, 20), 255, dtype=np.uint8)
    image[0:6, 0:6] = 0  # component A: 36 ink pixels, extent 6
    image[10:16, 10:16] = 0  # component B: 36 ink pixels, extent 6
    image[19, 19] = 0  # component C: 1 ink pixel, extent 1
    return image


def test_ink_coverage_on_blank_image_is_zero() -> None:
    blank = np.full((10, 10), 255, dtype=np.uint8)
    assert ink_coverage(blank) == 0.0


def test_ink_coverage_matches_known_fraction() -> None:
    image = _make_image()
    expected = (36 + 36 + 1) / (20 * 20)
    assert ink_coverage(image) == pytest.approx(expected)


def test_component_count_matches_known_components() -> None:
    image = _make_image()
    assert component_count(image) == 3


def test_component_count_on_blank_image_is_zero() -> None:
    blank = np.full((10, 10), 255, dtype=np.uint8)
    assert component_count(blank) == 0


def test_median_stroke_length_matches_known_extents() -> None:
    image = _make_image()
    # Extents are [6, 6, 1] -> median is 6.
    assert median_stroke_length(image) == pytest.approx(6.0)


def test_median_stroke_length_on_blank_image_is_zero() -> None:
    blank = np.full((10, 10), 255, dtype=np.uint8)
    assert median_stroke_length(blank) == 0.0


def test_noise_fraction_matches_known_small_component_share() -> None:
    image = _make_image()
    # Components with extent < 4: only component C (1 ink pixel).
    # Total ink pixels: 36 + 36 + 1 = 73.
    expected = 1 / 73
    assert noise_fraction(image, min_extent=4) == pytest.approx(expected)


def test_noise_fraction_is_zero_when_all_components_are_large() -> None:
    image = _make_image()
    assert noise_fraction(image, min_extent=1) == 0.0


def test_noise_fraction_on_blank_image_is_zero() -> None:
    blank = np.full((10, 10), 255, dtype=np.uint8)
    assert noise_fraction(blank) == 0.0


def test_compute_metrics_aggregates_all_four_values() -> None:
    image = _make_image()
    metrics = compute_metrics(image, min_extent=4)

    assert metrics.ink_coverage == pytest.approx(ink_coverage(image))
    assert metrics.component_count == component_count(image)
    assert metrics.median_stroke_length == pytest.approx(median_stroke_length(image))
    assert metrics.noise_fraction == pytest.approx(noise_fraction(image, min_extent=4))


def _make_boundary_gt() -> np.ndarray:
    """A 30x30 reference with a single ink pixel, for hand-computable matching."""
    image = np.full((30, 30), 255, dtype=np.uint8)
    image[15, 15] = 0
    return image


def test_boundary_f_measure_exact_match_is_perfect() -> None:
    gt = _make_boundary_gt()
    result = boundary_f_measure(gt, gt, tolerance=0)
    assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)
    assert result.f1_normalized == pytest.approx(1.0)


def test_boundary_f_measure_within_tolerance_still_matches() -> None:
    gt = _make_boundary_gt()
    pred = np.full((30, 30), 255, dtype=np.uint8)
    pred[15, 16] = 0  # 1px away from the single gt ink pixel

    result = boundary_f_measure(pred, gt, tolerance=2)
    assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)


def test_boundary_f_measure_outside_tolerance_scores_zero() -> None:
    gt = _make_boundary_gt()
    pred = np.full((30, 30), 255, dtype=np.uint8)
    pred[15, 16] = 0  # 1px away

    result = boundary_f_measure(pred, gt, tolerance=0)
    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)


def test_boundary_f_measure_on_two_blank_images_is_zero() -> None:
    blank = np.full((10, 10), 255, dtype=np.uint8)
    result = boundary_f_measure(blank, blank)
    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)
    assert result.f1_normalized == 0.0
    assert result.ink_admissible is False


def test_compare_boundaries_resizes_before_matching() -> None:
    # A 60x60 gt with a 4px-wide ink stripe, and a 30x30 pred with the
    # matching stripe at half resolution -- only fair to compare after
    # resizing both to a common long side.
    gt = np.full((60, 60), 255, dtype=np.uint8)
    gt[28:32, :] = 0
    pred = np.full((30, 30), 255, dtype=np.uint8)
    pred[14:16, :] = 0

    result = compare_boundaries(pred, gt, long_side=60, tolerance=2)
    assert result.precision > 0.9
    assert result.recall > 0.9
    assert result.f1 > 0.9


def test_compare_boundaries_default_tolerance_is_relative_to_long_side() -> None:
    gt = np.full((60, 60), 255, dtype=np.uint8)
    gt[28:32, :] = 0
    pred = gt.copy()

    result = compare_boundaries(pred, gt, long_side=60)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_boundary_f_measure_degenerate_baselines_lose_to_a_real_engine() -> None:
    """Guard against a metric that rewards drawing more ink instead of the right ink."""
    photo = make_synthetic_photo()
    gt = CannyEngine().convert(photo, line_thickness=1)
    pred = ChainedEngine().convert(photo, line_thickness=1)

    engine_f1 = boundary_f_measure(pred, gt, tolerance=2).f1

    all_ink = np.zeros_like(gt)
    black_f1 = boundary_f_measure(all_ink, gt, tolerance=2).f1

    grid = np.full_like(gt, 255)
    grid[::8, :] = 0
    grid[:, ::8] = 0
    grid_f1 = boundary_f_measure(grid, gt, tolerance=2).f1

    assert engine_f1 > black_f1 + 0.15
    assert engine_f1 > grid_f1 + 0.15
    assert black_f1 < 0.5
    assert grid_f1 < 0.5


def _make_large_reference(long_side: int = 864) -> np.ndarray:
    """A line-art-like reference at a realistic working scale.

    The 64x48 ``make_synthetic_photo`` fixture is far too small for
    pixel-scale calibration checks: a 3px shift is a huge fraction of a
    64px-wide image but a small one at the 864px long side the project's
    reference images actually use (matching ``docs/DIAGNOSIS.md``'s
    calibration table). This draws a handful of thick strokes -- closer to
    a real line drawing than a single dot -- on a canvas at that scale.
    """
    height, width = long_side * 3 // 4, long_side
    gt = np.full((height, width), 255, dtype=np.uint8)
    cv2.rectangle(gt, (width // 6, height // 6), (width // 2, height // 2), 0, thickness=3)
    cv2.circle(gt, (width * 3 // 4, height * 3 // 4), long_side // 8, 0, thickness=3)
    cv2.line(gt, (width // 10, height * 9 // 10), (width * 9 // 10, height * 9 // 10), 0, 3)
    return gt


def test_degenerate_baselines_stay_near_the_floor() -> None:
    """The metric this project uses must not reward information-free predictions.

    Per ``docs/DIAGNOSIS.md`` §0, these baselines (all-black page, regular
    grid, uniform noise) should score close to :func:`degenerate_floor`,
    not far above it -- confirming the floor is actually a floor, not just
    one arbitrary degenerate case among many that could score higher.
    """
    gt = _make_large_reference()
    tolerance = 4
    floor = degenerate_floor(gt, tolerance=tolerance)
    assert floor > 0.0  # sanity: the floor itself is non-trivial, not zero

    all_ink = np.zeros_like(gt)
    black = boundary_f_measure(all_ink, gt, tolerance=tolerance)

    rng = np.random.default_rng(0)
    noise = np.where(rng.random(gt.shape) < 0.04, 0, 255).astype(np.uint8)
    noisy = boundary_f_measure(noise, gt, tolerance=tolerance)

    grid = np.full_like(gt, 255)
    grid[::8, :] = 0
    grid[:, ::8] = 0
    gridded = boundary_f_measure(grid, gt, tolerance=tolerance)

    assert black.f1 <= floor + 0.05
    assert noisy.f1 <= floor + 0.05
    assert gridded.f1 <= floor + 0.05


def test_translated_reference_scores_well_above_the_floor() -> None:
    """A drawing translated by 3px is a stylistic/alignment difference, not noise.

    The pre-fix metric scored this *below* the plain `canny` engine at a
    fixed 2px tolerance -- exactly the bug this phase fixes (see
    ``docs/DIAGNOSIS.md`` §0's calibration table: 0.502 vs 0.614). At the
    relative tolerance derived from the image's own long side, it must
    score well above the degenerate floor instead.
    """
    gt = _make_large_reference()

    translated = np.full_like(gt, 255)
    translated[:, 3:] = gt[:, :-3]

    result = compare_boundaries(translated, gt, long_side=max(gt.shape))
    assert result.f1_normalized > 0.80


def test_dilated_reference_is_not_ink_admissible() -> None:
    gt = CannyEngine().convert(make_synthetic_photo(), line_thickness=1)
    ink_mask = (gt < 160).astype(np.uint8)
    dilated_mask = cv2.dilate(ink_mask, np.ones((3, 3), np.uint8))
    dilated = np.where(dilated_mask > 0, 0, 255).astype(np.uint8)

    result = boundary_f_measure(dilated, gt, tolerance=2)
    assert result.ink_admissible is False
