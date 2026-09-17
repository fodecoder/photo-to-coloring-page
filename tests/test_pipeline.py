"""Tests for the shared image pipeline (loading, resizing, saving)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coloring_page.pipeline import (
    UnsupportedFormatError,
    default_line_thickness,
    derive_kernel_size,
    load_image,
    remove_short_strokes,
    remove_small_specks,
    resize_to_max_dimension,
    save_image,
    smooth_preserving_edges,
)


def test_load_image_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "does_not_exist.png")


def test_load_image_unsupported_format_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not an image")

    with pytest.raises(UnsupportedFormatError):
        load_image(bogus)


def test_load_image_reads_valid_file(synthetic_photo_path: Path) -> None:
    image = load_image(synthetic_photo_path)
    assert image.ndim == 3
    assert image.shape[2] == 3


def test_resize_to_max_dimension_downscales(synthetic_photo: np.ndarray) -> None:
    resized = resize_to_max_dimension(synthetic_photo, max_dimension=32)
    assert max(resized.shape[:2]) <= 32


def test_resize_to_max_dimension_does_not_upscale(synthetic_photo: np.ndarray) -> None:
    resized = resize_to_max_dimension(synthetic_photo, max_dimension=10_000)
    assert resized.shape == synthetic_photo.shape


def test_resize_to_max_dimension_none_is_noop(synthetic_photo: np.ndarray) -> None:
    resized = resize_to_max_dimension(synthetic_photo, max_dimension=None)
    assert resized.shape == synthetic_photo.shape


def test_save_image_creates_parent_dirs(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "out.png"
    save_image(np.zeros((10, 10), dtype=np.uint8), destination)
    assert destination.exists()


def test_smooth_preserving_edges_keeps_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    gray = np.mean(synthetic_photo, axis=2).astype(np.uint8)
    smoothed = smooth_preserving_edges(gray)
    assert smoothed.shape == gray.shape
    assert smoothed.dtype == gray.dtype


def test_remove_short_strokes_drops_isolated_dot_keeps_large_shape() -> None:
    image = np.full((20, 20), 255, dtype=np.uint8)
    # A single-pixel noise speck, isolated from anything else.
    image[2, 2] = 0
    # A real stroke: a solid 6x6 block, well above the default min_extent.
    image[10:16, 10:16] = 0

    cleaned = remove_short_strokes(image, min_extent=4)

    assert cleaned[2, 2] == 255
    assert np.all(cleaned[10:16, 10:16] == 0)


def test_remove_short_strokes_keeps_thin_long_line_despite_small_area() -> None:
    image = np.full((30, 30), 255, dtype=np.uint8)
    # A 1px-wide, 20px-long line: tiny area (20px) but a large extent (20),
    # which is exactly the case an area-based filter gets wrong.
    image[15, 5:25] = 0

    cleaned = remove_short_strokes(image, min_extent=4)

    assert np.all(cleaned[15, 5:25] == 0)


def test_remove_short_strokes_drops_short_line_below_min_extent() -> None:
    image = np.full((30, 30), 255, dtype=np.uint8)
    # A 1px-wide, 3px-long line: extent 3, below a min_extent of 4.
    image[15, 5:8] = 0

    cleaned = remove_short_strokes(image, min_extent=4)

    assert np.all(cleaned[15, 5:8] == 255)


def test_derive_kernel_size_scales_with_working_dimension() -> None:
    small = derive_kernel_size(700, fraction=0.01)
    large = derive_kernel_size(1400, fraction=0.01)
    assert large >= small


def test_derive_kernel_size_respects_min_value() -> None:
    assert derive_kernel_size(10, fraction=0.01, min_value=5) == 5


def test_derive_kernel_size_forces_odd_when_requested() -> None:
    size = derive_kernel_size(1400, fraction=0.005, odd=True)
    assert size % 2 == 1


def test_derive_kernel_size_allows_even_when_not_requested() -> None:
    size = derive_kernel_size(1400, fraction=0.005714, min_value=3, odd=False)
    assert size == round(1400 * 0.005714)


def test_default_line_thickness_scales_with_working_dimension() -> None:
    assert default_line_thickness(1400) == max(2, round(1400 / 500))
    assert default_line_thickness(500) >= 2
    assert default_line_thickness(2000) > default_line_thickness(500)


def test_remove_small_specks_is_deprecated_alias() -> None:
    image = np.full((20, 20), 255, dtype=np.uint8)
    image[2, 2] = 0
    image[10:16, 10:16] = 0

    with pytest.warns(DeprecationWarning):
        legacy_result = remove_small_specks(image, min_area=4)

    assert np.array_equal(legacy_result, remove_short_strokes(image, min_extent=4))
