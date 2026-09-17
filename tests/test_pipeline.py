"""Tests for the shared image pipeline (loading, resizing, saving)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coloring_page.pipeline import (
    UnsupportedFormatError,
    load_image,
    resize_to_max_dimension,
    save_image,
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
