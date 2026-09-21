"""Tests for the adaptive-threshold conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.drawing import rasterize
from coloring_page.engines.adaptive import AdaptiveEngine


def test_requires_serial_execution_is_false() -> None:
    assert AdaptiveEngine.requires_serial_execution is False


def test_output_is_a_nonempty_drawing(synthetic_photo: np.ndarray) -> None:
    engine = AdaptiveEngine()
    drawing = engine.convert(synthetic_photo)

    assert len(drawing.paths) > 0
    assert drawing.aspect_ratio == synthetic_photo.shape[1] / synthetic_photo.shape[0]


def test_rasterized_output_is_mostly_near_black_or_white(synthetic_photo: np.ndarray) -> None:
    engine = AdaptiveEngine()
    drawing = engine.convert(synthetic_photo)
    result = rasterize(drawing, long_side_px=max(synthetic_photo.shape[:2]))

    near_extreme = (result < 10) | (result > 245)
    assert np.mean(near_extreme) > 0.9


def test_even_block_size_is_made_odd() -> None:
    engine = AdaptiveEngine(block_size=10)
    assert engine.block_size == 11


def test_isolated_noise_is_cleaned_up(
    synthetic_photo: np.ndarray, noisy_synthetic_photo: np.ndarray
) -> None:
    engine = AdaptiveEngine()
    long_side = max(synthetic_photo.shape[:2])
    clean_result = rasterize(engine.convert(synthetic_photo), long_side_px=long_side)
    noisy_result = rasterize(engine.convert(noisy_synthetic_photo), long_side_px=long_side)

    def count_small_components(binary: np.ndarray, max_area: int = 3) -> int:
        ink_mask = (binary < 128).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        return sum(
            1 for label in range(1, num_labels) if stats[label, cv2.CC_STAT_AREA] <= max_area
        )

    assert count_small_components(noisy_result) == count_small_components(clean_result)
