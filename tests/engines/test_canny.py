"""Tests for the Canny conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.drawing import rasterize
from coloring_page.engines.canny import CannyEngine


def test_requires_serial_execution_is_false() -> None:
    assert CannyEngine.requires_serial_execution is False


def test_output_is_a_nonempty_drawing(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    drawing = engine.convert(synthetic_photo)

    assert len(drawing.paths) > 0
    assert drawing.aspect_ratio == synthetic_photo.shape[1] / synthetic_photo.shape[0]


def test_rasterized_output_is_mostly_near_black_or_white(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    drawing = engine.convert(synthetic_photo)
    result = rasterize(drawing, long_side_px=max(synthetic_photo.shape[:2]))

    # rasterize() redraws traced paths with cv2.LINE_AA, so a handful of
    # anti-aliased pixels along each stroke's edge is expected; the vast
    # majority of pixels should still fall at the pure ink/background
    # extremes, unlike genuinely soft (unbinarized) shading.
    near_extreme = (result < 10) | (result > 245)
    assert np.mean(near_extreme) > 0.9


def test_higher_thickness_adds_more_ink(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    drawing = engine.convert(synthetic_photo)
    long_side = max(synthetic_photo.shape[:2])
    thin = rasterize(drawing, long_side_px=long_side, line_thickness=1)
    thick = rasterize(drawing, long_side_px=long_side, line_thickness=3)

    thin_ink = np.sum(thin < 128)
    thick_ink = np.sum(thick < 128)
    assert thick_ink >= thin_ink


def test_explicit_thresholds_bypass_auto_gradient_percentile(synthetic_photo: np.ndarray) -> None:
    auto_engine = CannyEngine()
    explicit_engine = CannyEngine(low_threshold=10, high_threshold=20)

    auto_drawing = auto_engine.convert(synthetic_photo)
    explicit_drawing = explicit_engine.convert(synthetic_photo)

    # Very low explicit thresholds pick up far more edges than the
    # gradient-percentile auto mode would on this image, so the two
    # outputs should differ -- proving the explicit values were actually
    # used rather than silently overridden by the auto heuristic.
    assert len(auto_drawing.paths) != len(explicit_drawing.paths)


def test_ink_coverage_in_reasonable_band() -> None:
    # Regression test: the previous median-intensity auto-threshold
    # heuristic produced badly broken contours on light backgrounds.
    # This band is wide on purpose -- it exists to catch a silent total
    # failure, not to pin an exact tuning target.
    image = np.full((96, 96, 3), 250, dtype=np.uint8)
    image[30:36, 24:72] = 15

    engine = CannyEngine()
    drawing = engine.convert(image)
    result = rasterize(drawing, long_side_px=max(image.shape[:2]))

    ink_fraction = float(np.mean(result < 128))
    assert 0.01 <= ink_fraction <= 0.30


def test_isolated_noise_is_cleaned_up(
    synthetic_photo: np.ndarray, noisy_synthetic_photo: np.ndarray
) -> None:
    engine = CannyEngine()
    long_side = max(synthetic_photo.shape[:2])
    clean_result = rasterize(engine.convert(synthetic_photo), long_side_px=long_side)
    noisy_result = rasterize(engine.convert(noisy_synthetic_photo), long_side_px=long_side)

    def count_small_components(binary: np.ndarray, max_area: int = 3) -> int:
        ink_mask = (binary < 128).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        return sum(
            1 for label in range(1, num_labels) if stats[label, cv2.CC_STAT_AREA] <= max_area
        )

    # Injected salt-and-pepper noise should not survive as tiny isolated
    # ink specks once the engine's speckle cleanup runs.
    assert count_small_components(noisy_result) == count_small_components(clean_result)
