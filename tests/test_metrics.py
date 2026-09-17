"""Tests for the line-art quality metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from coloring_page.metrics import (
    component_count,
    compute_metrics,
    ink_coverage,
    median_stroke_length,
    noise_fraction,
)


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
