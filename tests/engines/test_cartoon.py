"""Tests for the color-quantization cartoon conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.cartoon import CartoonEngine


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = CartoonEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = CartoonEngine()
    result = engine.convert(synthetic_photo)

    # The synthetic photo is mostly flat/gradient area with a couple of
    # shapes; quantized-region boundaries should stay a minority of pixels.
    assert np.mean(result) > 127


def test_isolated_noise_is_cleaned_up(
    synthetic_photo: np.ndarray, noisy_synthetic_photo: np.ndarray
) -> None:
    engine = CartoonEngine()
    clean_result = engine.convert(synthetic_photo)
    noisy_result = engine.convert(noisy_synthetic_photo)

    def count_small_components(binary: np.ndarray, max_area: int = 3) -> int:
        ink_mask = (binary < 128).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        return sum(
            1 for label in range(1, num_labels) if stats[label, cv2.CC_STAT_AREA] <= max_area
        )

    assert count_small_components(noisy_result) == count_small_components(clean_result)
