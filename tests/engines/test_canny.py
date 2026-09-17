"""Tests for the Canny conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.canny import CannyEngine


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_is_binary_line_art(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    result = engine.convert(synthetic_photo)

    # Canny output is strictly black-or-white; inverted it stays that way.
    assert set(np.unique(result).tolist()) <= {0, 255}


def test_higher_thickness_adds_more_ink(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    thin = engine.convert(synthetic_photo, line_thickness=1)
    thick = engine.convert(synthetic_photo, line_thickness=3)

    thin_ink = np.sum(thin < 128)
    thick_ink = np.sum(thick < 128)
    assert thick_ink >= thin_ink


def test_explicit_thresholds_bypass_auto_gradient_percentile(synthetic_photo: np.ndarray) -> None:
    auto_engine = CannyEngine()
    explicit_engine = CannyEngine(low_threshold=10, high_threshold=20)

    auto_result = auto_engine.convert(synthetic_photo)
    explicit_result = explicit_engine.convert(synthetic_photo)

    # Very low explicit thresholds pick up far more edges than the
    # gradient-percentile auto mode would on this image, so the two
    # outputs should differ -- proving the explicit values were actually
    # used rather than silently overridden by the auto heuristic.
    assert not np.array_equal(auto_result, explicit_result)


def test_ink_coverage_in_reasonable_band() -> None:
    # Regression test: the previous median-intensity auto-threshold
    # heuristic produced badly broken contours on light backgrounds.
    # This band is wide on purpose -- it exists to catch a silent total
    # failure, not to pin an exact tuning target.
    image = np.full((96, 96, 3), 250, dtype=np.uint8)
    image[30:36, 24:72] = 15

    engine = CannyEngine()
    result = engine.convert(image)

    ink_fraction = float(np.mean(result < 128))
    assert 0.01 <= ink_fraction <= 0.30


def test_isolated_noise_is_cleaned_up(
    synthetic_photo: np.ndarray, noisy_synthetic_photo: np.ndarray
) -> None:
    engine = CannyEngine()
    clean_result = engine.convert(synthetic_photo)
    noisy_result = engine.convert(noisy_synthetic_photo)

    def count_small_components(binary: np.ndarray, max_area: int = 3) -> int:
        ink_mask = (binary < 128).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        return sum(
            1 for label in range(1, num_labels) if stats[label, cv2.CC_STAT_AREA] <= max_area
        )

    # Injected salt-and-pepper noise should not survive as tiny isolated
    # ink specks once the engine's speckle cleanup runs.
    assert count_small_components(noisy_result) == count_small_components(clean_result)
