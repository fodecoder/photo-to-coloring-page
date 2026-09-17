"""Tests for the XDoG conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.xdog import XDoGEngine


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = XDoGEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = XDoGEngine()
    result = engine.convert(synthetic_photo)

    # Most of the synthetic photo is flat/gradient area with no edges,
    # so the majority of output pixels should be near-white background.
    assert np.mean(result) > 127


def test_isolated_noise_is_cleaned_up(
    synthetic_photo: np.ndarray, noisy_synthetic_photo: np.ndarray
) -> None:
    engine = XDoGEngine()
    clean_result = engine.convert(synthetic_photo)
    noisy_result = engine.convert(noisy_synthetic_photo)

    def count_small_components(binary: np.ndarray, max_area: int = 3) -> int:
        ink_mask = (binary < 128).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        return sum(
            1 for label in range(1, num_labels) if stats[label, cv2.CC_STAT_AREA] <= max_area
        )

    assert count_small_components(noisy_result) == count_small_components(clean_result)
