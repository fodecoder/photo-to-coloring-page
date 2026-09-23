"""Tests for the shared image pipeline (loading, resizing, saving)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coloring_page.drawing import Drawing
from coloring_page.drawing import Path as VectorPath
from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.pipeline import (
    UnsupportedFormatError,
    default_line_thickness,
    derive_kernel_size,
    l0_smooth,
    load_image,
    resize_to_max_dimension,
    run_pipeline,
    save_image,
    smooth_preserving_edges,
)


class _StubDebugEngine(ConversionEngine):
    """Minimal engine that reports fixed debug stages, for testing DebugSink wiring in isolation."""

    name = "stub-debug"

    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> Drawing:
        gray = np.full(image.shape[:2], 255, dtype=np.uint8)
        if debug is not None:
            debug.save("stage_one", gray)
            debug.save("stage_two", gray)
        path = VectorPath(
            points=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            closed=False,
            kind="boundary",
        )
        return Drawing(paths=(path,), aspect_ratio=image.shape[1] / image.shape[0])


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


def test_default_line_thickness_matches_expected_values_at_key_resolutions() -> None:
    # 700px -> 1px, 1400px -> 2px, 2100px -> 3px: ~0.15% of the long side.
    assert default_line_thickness(700) == 1
    assert default_line_thickness(1400) == 2
    assert default_line_thickness(2100) == 3


def test_default_line_thickness_never_goes_below_one() -> None:
    assert default_line_thickness(1) >= 1


def test_default_line_thickness_scales_with_working_dimension() -> None:
    assert default_line_thickness(2000) > default_line_thickness(500)


def test_run_pipeline_writes_debug_stages_when_debug_dir_given(
    tmp_path: Path, synthetic_photo: np.ndarray
) -> None:
    debug_dir = tmp_path / "debug"

    run_pipeline(synthetic_photo, _StubDebugEngine(), debug_dir=debug_dir)

    saved = sorted(p.name for p in debug_dir.iterdir())
    assert saved == ["01_stage_one.png", "02_stage_two.png"]


def test_run_pipeline_skips_debug_output_when_debug_dir_omitted(
    tmp_path: Path, synthetic_photo: np.ndarray
) -> None:
    run_pipeline(synthetic_photo, _StubDebugEngine())

    # No debug_dir was created anywhere under tmp_path.
    assert not any(tmp_path.iterdir())


def test_l0_smooth_keeps_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    result = l0_smooth(synthetic_photo, lambda_=0.02, kappa=2.0)
    assert result.shape == synthetic_photo.shape
    assert result.dtype == synthetic_photo.dtype


def test_l0_smooth_does_not_distort_border_of_mismatched_edges() -> None:
    # A left-to-right ramp has opposite edges 140 levels apart. Raw
    # cv2.ximgproc.l0Smooth wraps them into each other (periodic FFT
    # boundary) and shifts the border columns by ~70 levels; the padded
    # version should keep them no more distorted than the interior.
    ramp = np.tile(np.linspace(60, 200, 300).astype(np.uint8), (200, 1))
    image = np.dstack([ramp, ramp, ramp])

    result = l0_smooth(image, lambda_=0.02, kappa=2.0)

    deviation = np.abs(result.astype(np.int16) - image.astype(np.int16))
    assert deviation[:, [0, -1]].max() <= 20
