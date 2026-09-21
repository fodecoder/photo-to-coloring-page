"""Tests for the XDoG conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.drawing import rasterize
from coloring_page.engines.xdog import XDoGEngine
from coloring_page.pipeline import default_line_thickness


def test_requires_serial_execution_is_false() -> None:
    assert XDoGEngine.requires_serial_execution is False


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


def test_output_is_a_nonempty_drawing(synthetic_photo: np.ndarray) -> None:
    engine = XDoGEngine()
    drawing = engine.convert(synthetic_photo)

    assert len(drawing.paths) > 0
    assert drawing.aspect_ratio == synthetic_photo.shape[1] / synthetic_photo.shape[0]


def test_output_has_light_background() -> None:
    photo = _make_page_like_photo()
    engine = XDoGEngine()
    drawing = engine.convert(photo)
    result = rasterize(drawing, long_side_px=max(photo.shape[:2]))

    # A page-like input (mostly light background, one dark stroke) should
    # produce output that is still mostly white background.
    assert np.mean(result) > 127


def test_ink_coverage_in_reasonable_band() -> None:
    # Regression test: the previous inverted-sign XDoG formula collapsed
    # to ~0.05% ink coverage (essentially a blank page) on real photos --
    # this exists to catch that kind of near-total collapse, not to pin
    # an exact tuning target. The floor is lower than it used to be:
    # paths_from_mask traces a single 1px centerline per edge instead of
    # keeping the raw threshold mask's full-width ink band, so a small
    # single-stroke input legitimately has much less ink now than before
    # this engine was vectorized -- that reduction is the intended effect
    # of tracing a centerline instead of a raw mask, not a regression.
    photo = _make_page_like_photo()
    engine = XDoGEngine()
    drawing = engine.convert(photo)
    long_side = max(photo.shape[:2])
    result = rasterize(
        drawing, long_side_px=long_side, line_thickness=default_line_thickness(long_side)
    )

    ink_fraction = float(np.mean(result < 128))
    assert 0.002 <= ink_fraction <= 0.30


def test_isolated_noise_is_cleaned_up(
    synthetic_photo: np.ndarray, noisy_synthetic_photo: np.ndarray
) -> None:
    engine = XDoGEngine()
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
