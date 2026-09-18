"""Tests for the flatten-skeletonize-redraw (skeleton) conversion engine."""

from __future__ import annotations

import numpy as np

from coloring_page.engines.skeleton_redraw import SkeletonRedrawEngine
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


# prune_short_branches itself is now shared in coloring_page.postprocess
# and tested in tests/test_postprocess.py; this engine only exercises it
# as one stage of its own pipeline (test_debug_sink_receives_all_four_stages
# above already confirms it runs).
