"""Tests for the region-boundary (region) conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.region import (
    RegionEngine,
    _label_boundaries,
    _labels_from_colors,
    filter_by_contrast,
    gradient_boundaries,
    segment_mean_shift,
    segment_superpixels,
)
from coloring_page.metrics import ink_coverage


def _make_two_color_blocks() -> np.ndarray:
    """A 60x60 BGR image with two flat-colored halves, for hand-checkable labels."""
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    image[:, :30] = (20, 20, 20)
    image[:, 30:] = (220, 220, 220)
    return image


def test_labels_from_colors_groups_identical_colors() -> None:
    image = _make_two_color_blocks()
    labels = _labels_from_colors(image, quantize_step=4)

    assert labels[0, 0] == labels[59, 29]  # same block, opposite corners
    assert labels[0, 0] != labels[0, 59]  # different blocks


def test_label_boundaries_marks_only_the_seam() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[:, 5:] = 1

    boundary = _label_boundaries(labels)

    assert boundary[3, 4] and boundary[3, 5]  # both sides of the seam
    assert not boundary[3, 0]  # far from the seam
    assert not boundary[3, 9]


def test_filter_by_contrast_keeps_high_contrast_boundary() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[:, 5:] = 1
    boundary = _label_boundaries(labels)
    lab = np.zeros((10, 10, 3), dtype=np.uint8)
    lab[:, 5:] = 200  # a large color difference between the two labels

    kept = filter_by_contrast(boundary, labels, lab, min_contrast=50.0)

    assert kept.any()


def test_filter_by_contrast_drops_low_contrast_boundary() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[:, 5:] = 1
    boundary = _label_boundaries(labels)
    lab = np.zeros((10, 10, 3), dtype=np.uint8)
    lab[:, 5:] = 1  # two labels, near-identical color -- a segmentation artifact

    kept = filter_by_contrast(boundary, labels, lab, min_contrast=50.0)

    assert not kept.any()


def test_gradient_boundaries_finds_the_seam() -> None:
    image = _make_two_color_blocks()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    boundary = gradient_boundaries(lab, percentile=90.0)

    assert boundary[:, 28:32].any()


def test_segment_mean_shift_separates_two_flat_blocks() -> None:
    image = _make_two_color_blocks()
    labels = segment_mean_shift(image, sp=10, sr=20)

    assert labels[30, 5] != labels[30, 55]


def test_segment_superpixels_produces_multiple_labels() -> None:
    image = _make_two_color_blocks()
    labels = segment_superpixels(image, algorithm="seeds", num_superpixels=20)

    assert labels.shape == image.shape[:2]
    assert len(np.unique(labels)) > 1


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = RegionEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = RegionEngine()
    result = engine.convert(synthetic_photo)

    assert np.mean(result) > 127


def test_ink_coverage_in_reasonable_band(synthetic_photo: np.ndarray) -> None:
    # Regression band, not a tuning target: catches a total-failure
    # collapse (no surviving boundaries, or everything redrawn as ink).
    engine = RegionEngine()
    result = engine.convert(synthetic_photo)

    assert 0.001 <= ink_coverage(result) <= 0.40


def test_higher_thickness_adds_more_ink(synthetic_photo: np.ndarray) -> None:
    engine = RegionEngine()
    thin = engine.convert(synthetic_photo, line_thickness=1)
    thick = engine.convert(synthetic_photo, line_thickness=3)

    assert np.sum(thick < 128) >= np.sum(thin < 128)


def test_superpixel_segmentation_runs_end_to_end(synthetic_photo: np.ndarray) -> None:
    engine = RegionEngine(segmentation="seeds")
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]


def test_gradient_boundary_detection_runs_end_to_end(synthetic_photo: np.ndarray) -> None:
    engine = RegionEngine(boundary_detection="gradient")
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]


def test_unknown_boundary_detection_rejected(synthetic_photo: np.ndarray) -> None:
    import pytest

    engine = RegionEngine(boundary_detection="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown boundary_detection"):
        engine.convert(synthetic_photo)


def test_min_contrast_affects_output(synthetic_photo: np.ndarray) -> None:
    permissive = RegionEngine(min_contrast=0.0).convert(synthetic_photo)
    strict = RegionEngine(min_contrast=100.0).convert(synthetic_photo)

    assert not np.array_equal(permissive, strict)


class _RecordingDebugSink:
    """Structurally satisfies the DebugSink protocol (matching .save signature)."""

    def __init__(self) -> None:
        self.stage_names: list[str] = []

    def save(self, stage_name: str, image: np.ndarray) -> None:
        self.stage_names.append(stage_name)


def test_debug_sink_receives_all_five_stages(synthetic_photo: np.ndarray) -> None:
    sink = _RecordingDebugSink()
    engine = RegionEngine()

    engine.convert(synthetic_photo, debug=sink)

    assert sink.stage_names == [
        "flattened",
        "segments",
        "boundaries_raw",
        "boundaries_filtered",
        "redraw",
    ]


def test_registered_in_engines() -> None:
    from coloring_page.engines.registry import ENGINES

    assert "region" in ENGINES
