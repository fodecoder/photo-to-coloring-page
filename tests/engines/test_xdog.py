"""Tests for the XDoG conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.xdog import XDoGEngine


def _make_page_like_photo(width: int = 96, height: int = 96) -> np.ndarray:
    """A mostly-light synthetic "page photo": light background, one dark stroke.

    XDoG's epsilon threshold (0.95) is calibrated for the real target
    domain -- a photographed illustration page, which is predominantly
    light paper with comparatively little dark content -- not for the
    shared ``synthetic_photo`` fixture, which is a full 0-255 tonal
    gradient covering every intensity equally. On that full-range
    gradient, roughly half the pixels are legitimately darker than the
    threshold and correctly render as ink, which is correct behavior for
    that input but not representative of "mostly-white page" ink
    coverage. This helper stays local to XDoG's own tests so it doesn't
    change the shared fixture other engines' tests also rely on.
    """
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    image[height // 3 : height // 3 + 6, width // 4 : width * 3 // 4] = 15
    return image


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = XDoGEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_has_light_background() -> None:
    engine = XDoGEngine()
    result = engine.convert(_make_page_like_photo())

    # A page-like input (mostly light background, one dark stroke) should
    # produce output that is still mostly white background.
    assert np.mean(result) > 127


def test_ink_coverage_in_reasonable_band() -> None:
    # Regression test: the previous inverted-sign XDoG formula collapsed
    # to ~0.05% ink coverage (essentially a blank page) on real photos.
    # This band is wide on purpose -- it exists to catch a silent total
    # failure like that, not to pin an exact tuning target.
    engine = XDoGEngine()
    result = engine.convert(_make_page_like_photo())

    ink_fraction = float(np.mean(result < 128))
    assert 0.01 <= ink_fraction <= 0.30


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
