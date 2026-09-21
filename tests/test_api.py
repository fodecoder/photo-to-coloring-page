"""Tests for the public ``convert_image`` entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coloring_page.api import convert_image
from coloring_page.exceptions import ImageTooLargeError, UnsupportedImageError
from coloring_page.profile import Profile


class TestConvertImageFromPath:
    def test_returns_a_result(self, synthetic_photo_path: Path) -> None:
        result = convert_image(synthetic_photo_path, profile=Profile(style="canny"))
        assert len(result.drawing.paths) >= 0
        assert result.report is not None

    def test_missing_path_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_image(tmp_path / "missing.png", profile=Profile())


class TestConvertImageFromNdarray:
    def test_accepts_in_memory_array(self, synthetic_photo: np.ndarray) -> None:
        result = convert_image(synthetic_photo, profile=Profile(style="canny"))
        assert result.drawing is not None

    def test_rejects_wrong_dtype(self, synthetic_photo: np.ndarray) -> None:
        bad = synthetic_photo.astype(np.float32)
        with pytest.raises(UnsupportedImageError):
            convert_image(bad, profile=Profile())

    def test_rejects_wrong_channel_count(self) -> None:
        bad = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(UnsupportedImageError):
            convert_image(bad, profile=Profile())


class TestConvertImageTimings:
    def test_timings_include_every_stage_and_total(self, synthetic_photo: np.ndarray) -> None:
        result = convert_image(synthetic_photo, profile=Profile(style="canny"))
        expected_stages = {
            "seed",
            "load",
            "size_check",
            "resize",
            "engine",
            "detail_filter",
            "validate",
            "total",
        }
        assert expected_stages.issubset(result.timings.keys())
        for value in result.timings.values():
            assert value >= 0


class TestConvertImageMaxPixels:
    def test_raises_when_image_exceeds_max_pixels(self, synthetic_photo: np.ndarray) -> None:
        pixel_count = synthetic_photo.shape[0] * synthetic_photo.shape[1]
        with pytest.raises(ImageTooLargeError):
            convert_image(
                synthetic_photo, profile=Profile(max_pixels=pixel_count - 1, style="canny")
            )

    def test_accepts_image_within_max_pixels(self, synthetic_photo: np.ndarray) -> None:
        pixel_count = synthetic_photo.shape[0] * synthetic_photo.shape[1]
        result = convert_image(
            synthetic_photo, profile=Profile(max_pixels=pixel_count, style="canny")
        )
        assert result is not None


class TestConvertImageDebugDir:
    def test_writes_pre_and_post_filter_drawings(
        self, tmp_path: Path, synthetic_photo: np.ndarray
    ) -> None:
        debug_dir = tmp_path / "debug"
        convert_image(synthetic_photo, profile=Profile(style="canny", debug_dir=debug_dir))

        saved = {p.name for p in debug_dir.iterdir()}
        assert any("drawing_pre_filter" in name for name in saved)
        assert any("drawing_post_filter" in name for name in saved)
