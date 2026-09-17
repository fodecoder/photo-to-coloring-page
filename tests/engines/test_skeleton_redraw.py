"""Tests for the flatten-skeletonize-redraw (skeleton) conversion engine."""

from __future__ import annotations

import numpy as np

from coloring_page.engines.skeleton_redraw import SkeletonRedrawEngine, _prune_short_branches
from coloring_page.metrics import ink_coverage


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = SkeletonRedrawEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = SkeletonRedrawEngine()
    result = engine.convert(synthetic_photo)

    assert np.mean(result) > 127


def test_ink_coverage_in_reasonable_band(synthetic_photo: np.ndarray) -> None:
    # Regression band, not a tuning target: catches a total-failure
    # collapse (empty skeleton, or everything redrawn as ink).
    engine = SkeletonRedrawEngine()
    result = engine.convert(synthetic_photo)

    assert 0.005 <= ink_coverage(result) <= 0.40


def test_higher_thickness_adds_more_ink(synthetic_photo: np.ndarray) -> None:
    engine = SkeletonRedrawEngine()
    thin = engine.convert(synthetic_photo, line_thickness=1)
    thick = engine.convert(synthetic_photo, line_thickness=3)

    assert np.sum(thick < 128) >= np.sum(thin < 128)


class _RecordingDebugSink:
    """Structurally satisfies the DebugSink protocol (matching .save signature)."""

    def __init__(self) -> None:
        self.stage_names: list[str] = []

    def save(self, stage_name: str, image: np.ndarray) -> None:
        self.stage_names.append(stage_name)


def test_debug_sink_receives_all_four_stages(synthetic_photo: np.ndarray) -> None:
    sink = _RecordingDebugSink()
    engine = SkeletonRedrawEngine()

    engine.convert(synthetic_photo, debug=sink)

    assert sink.stage_names == ["flattened", "edges", "skeleton", "redraw"]


def test_prune_short_branches_removes_short_dead_end_spur() -> None:
    skeleton = np.zeros((20, 30), dtype=np.uint8)
    skeleton[10, 0:30] = 255  # a long main line
    skeleton[10:14, 15] = 255  # a short (length-4) dead-end spur off it

    pruned = _prune_short_branches(skeleton, min_branch_length=6)

    assert np.all(pruned[12:14, 15] == 0)
    assert np.all(pruned[10, 0:30] == 255)


def test_prune_short_branches_keeps_branch_at_or_above_min_length() -> None:
    skeleton = np.zeros((25, 30), dtype=np.uint8)
    skeleton[10, 0:30] = 255  # a long main line
    skeleton[10:20, 15] = 255  # a long (length-10) dead-end spur off it

    pruned = _prune_short_branches(skeleton, min_branch_length=6)

    assert np.all(pruned[15:20, 15] == 255)


def test_prune_short_branches_on_blank_skeleton_is_noop() -> None:
    skeleton = np.zeros((10, 10), dtype=np.uint8)

    pruned = _prune_short_branches(skeleton, min_branch_length=6)

    assert np.all(pruned == 0)
