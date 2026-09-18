"""Tests for the shared soft-mask-to-line-art postprocessing module."""

from __future__ import annotations

import numpy as np
import pytest

from coloring_page.postprocess import (
    hysteresis_centerline,
    hysteresis_threshold,
    nms_centerline,
    normalize_percentile,
    prune_short_branches,
    redraw_centerline,
    soft_map_to_line_art,
)


def test_normalize_percentile_stretches_to_full_range() -> None:
    gray = np.full((20, 20), 128, dtype=np.uint8)
    gray[0, 0] = 100  # near the low end
    gray[0, 1] = 160  # near the high end

    stretched = normalize_percentile(gray, low_pct=0.0, high_pct=100.0)

    assert stretched.min() == 0
    assert stretched.max() == 255


def test_normalize_percentile_constant_image_does_not_divide_by_zero() -> None:
    gray = np.full((10, 10), 100, dtype=np.uint8)
    stretched = normalize_percentile(gray)
    assert stretched.dtype == np.uint8


def test_hysteresis_threshold_keeps_weak_ink_connected_to_strong_ink() -> None:
    gray = np.full((30, 30), 240, dtype=np.uint8)
    gray[10, 10:20] = 5  # strong stroke
    gray[10, 20:24] = 90  # weak continuation, directly connected

    result = hysteresis_threshold(gray, strong_threshold=20.0, weak_threshold=100.0)

    assert np.all(result[10, 10:20] == 0)
    assert np.all(result[10, 20:24] == 0)


def test_hysteresis_threshold_drops_weak_ink_not_connected_to_strong_ink() -> None:
    gray = np.full((30, 30), 240, dtype=np.uint8)
    gray[5, 5:15] = 5  # strong stroke
    gray[25, 25] = 90  # faint, disconnected speck

    result = hysteresis_threshold(gray, strong_threshold=20.0, weak_threshold=100.0)

    assert np.all(result[5, 5:15] == 0)
    assert result[25, 25] == 255


def test_prune_short_branches_removes_short_dead_end_spur() -> None:
    skeleton = np.zeros((20, 30), dtype=np.uint8)
    skeleton[10, 0:30] = 255  # a long main line
    skeleton[10:14, 15] = 255  # a short (length-4) dead-end spur off it

    pruned = prune_short_branches(skeleton, min_branch_length=6)

    assert np.all(pruned[12:14, 15] == 0)
    assert np.all(pruned[10, 0:30] == 255)


def test_prune_short_branches_keeps_branch_at_or_above_min_length() -> None:
    skeleton = np.zeros((25, 30), dtype=np.uint8)
    skeleton[10, 0:30] = 255  # a long main line
    skeleton[10:20, 15] = 255  # a long (length-10) dead-end spur off it

    pruned = prune_short_branches(skeleton, min_branch_length=6)

    assert np.all(pruned[15:20, 15] == 255)


def test_prune_short_branches_on_blank_skeleton_is_noop() -> None:
    skeleton = np.zeros((10, 10), dtype=np.uint8)
    pruned = prune_short_branches(skeleton, min_branch_length=6)
    assert np.all(pruned == 0)


def _make_wide_blob(width: int = 10) -> np.ndarray:
    """A soft map with one wide dark blob on a light background."""
    gray = np.full((60, 60), 220, dtype=np.uint8)
    gray[20:40, 25 : 25 + width] = 20
    return gray


def test_hysteresis_centerline_collapses_wide_blob_to_thin_line() -> None:
    gray = normalize_percentile(_make_wide_blob(width=10))

    centerline = hysteresis_centerline(gray)

    # A naive threshold would keep all 10 columns of the blob as ink;
    # the centerline should be far thinner -- this is the whole point of
    # extracting a centerline instead of thresholding directly.
    row = centerline[30, 20:40]
    assert 0 < np.count_nonzero(row) <= 3


def test_nms_centerline_collapses_wide_blob_to_thin_line() -> None:
    gray = normalize_percentile(_make_wide_blob(width=10))

    centerline = nms_centerline(gray)

    row = centerline[30, 20:40]
    assert 0 < np.count_nonzero(row) <= 3


def test_redraw_centerline_produces_project_convention_output() -> None:
    centerline = np.zeros((30, 30), dtype=np.uint8)
    centerline[15, 5:25] = 255

    result = redraw_centerline(centerline)

    assert result.shape == centerline.shape
    assert result.dtype == np.uint8
    # Project convention: 255 = background, lower = ink; a straight line
    # of centerline pixels should redraw as ink somewhere along row 15.
    assert np.any(result[15, :] < 128)
    assert np.mean(result) > 200  # mostly background


def test_redraw_centerline_on_blank_input_is_all_background() -> None:
    centerline = np.zeros((20, 20), dtype=np.uint8)
    result = redraw_centerline(centerline)
    assert np.all(result == 255)


@pytest.mark.parametrize("strategy", ["hysteresis", "nms"])
def test_soft_map_to_line_art_collapses_wide_blob(strategy: str) -> None:
    gray = _make_wide_blob(width=12)

    result = soft_map_to_line_art(gray, strategy=strategy)

    assert result.shape == gray.shape
    assert result.dtype == np.uint8
    # The wide blob should become a thin line, not stay a solid block:
    # far less than the original blob's ink coverage should survive.
    original_ink_fraction = np.mean(gray < 128)
    result_ink_fraction = np.mean(result < 128)
    assert result_ink_fraction < original_ink_fraction * 0.5


def test_soft_map_to_line_art_rejects_unknown_strategy() -> None:
    gray = _make_wide_blob()
    with pytest.raises(ValueError, match="Unknown strategy"):
        soft_map_to_line_art(gray, strategy="bogus")  # type: ignore[arg-type]


def test_soft_map_to_line_art_on_blank_input_is_all_background() -> None:
    gray = np.full((30, 30), 240, dtype=np.uint8)
    result = soft_map_to_line_art(gray)
    assert np.all(result == 255)
