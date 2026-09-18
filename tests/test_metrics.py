"""Tests for the line-art quality metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from coloring_page.engines.canny import CannyEngine
from coloring_page.engines.chained import ChainedEngine
from coloring_page.metrics import (
    boundary_f_measure,
    compare_boundaries,
    component_count,
    compute_metrics,
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
    precision, recall, f1 = boundary_f_measure(gt, gt, tolerance=0)
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)


def test_boundary_f_measure_within_tolerance_still_matches() -> None:
    gt = _make_boundary_gt()
    pred = np.full((30, 30), 255, dtype=np.uint8)
    pred[15, 16] = 0  # 1px away from the single gt ink pixel

    precision, recall, f1 = boundary_f_measure(pred, gt, tolerance=2)
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)


def test_boundary_f_measure_outside_tolerance_scores_zero() -> None:
    gt = _make_boundary_gt()
    pred = np.full((30, 30), 255, dtype=np.uint8)
    pred[15, 16] = 0  # 1px away

    precision, recall, f1 = boundary_f_measure(pred, gt, tolerance=0)
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_boundary_f_measure_on_two_blank_images_is_zero() -> None:
    blank = np.full((10, 10), 255, dtype=np.uint8)
    assert boundary_f_measure(blank, blank) == (0.0, 0.0, 0.0)


def test_compare_boundaries_resizes_before_matching() -> None:
    # A 60x60 gt with a 4px-wide ink stripe, and a 30x30 pred with the
    # matching stripe at half resolution -- only fair to compare after
    # resizing both to a common long side.
    gt = np.full((60, 60), 255, dtype=np.uint8)
    gt[28:32, :] = 0
    pred = np.full((30, 30), 255, dtype=np.uint8)
    pred[14:16, :] = 0

    precision, recall, f1 = compare_boundaries(pred, gt, long_side=60, tolerance=2)
    assert precision > 0.9
    assert recall > 0.9
    assert f1 > 0.9


def test_boundary_f_measure_degenerate_baselines_lose_to_a_real_engine() -> None:
    """Guard against a metric that rewards drawing more ink instead of the right ink."""
    photo = make_synthetic_photo()
    gt = CannyEngine().convert(photo, line_thickness=1)
    pred = ChainedEngine().convert(photo, line_thickness=1)

    _, _, engine_f1 = boundary_f_measure(pred, gt, tolerance=2)

    all_ink = np.zeros_like(gt)
    _, _, black_f1 = boundary_f_measure(all_ink, gt, tolerance=2)

    grid = np.full_like(gt, 255)
    grid[::8, :] = 0
    grid[:, ::8] = 0
    _, _, grid_f1 = boundary_f_measure(grid, gt, tolerance=2)

    assert engine_f1 > black_f1 + 0.15
    assert engine_f1 > grid_f1 + 0.15
    assert black_f1 < 0.5
    assert grid_f1 < 0.5
