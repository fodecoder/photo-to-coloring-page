"""Tests for the mean-shift segmentation cartoon conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.drawing import rasterize
from coloring_page.engines.cartoon import CartoonEngine


def test_requires_serial_execution_is_false() -> None:
    assert CartoonEngine.requires_serial_execution is False


def test_output_is_a_nonempty_drawing(synthetic_photo: np.ndarray) -> None:
    engine = CartoonEngine()
    drawing = engine.convert(synthetic_photo)

    assert len(drawing.paths) > 0
    assert drawing.aspect_ratio == synthetic_photo.shape[1] / synthetic_photo.shape[0]


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = CartoonEngine()
    drawing = engine.convert(synthetic_photo)
    result = rasterize(drawing, long_side_px=max(synthetic_photo.shape[:2]))

    # The synthetic photo is mostly flat/gradient area with a couple of
    # shapes; quantized-region boundaries should stay a minority of pixels.
    assert np.mean(result) > 127


def test_isolated_noise_is_cleaned_up(
    synthetic_photo: np.ndarray, noisy_synthetic_photo: np.ndarray
) -> None:
    engine = CartoonEngine()
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


def test_segmentation_parameters_affect_output(synthetic_photo: np.ndarray) -> None:
    detailed = CartoonEngine(spatial_radius=5, color_radius=15).convert(synthetic_photo)
    coarse = CartoonEngine(spatial_radius=40, color_radius=80).convert(synthetic_photo)

    assert len(detailed.paths) != len(coarse.paths)
