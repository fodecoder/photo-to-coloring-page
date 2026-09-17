"""Tests for the flatten-chain-redraw (chained) conversion engine."""

from __future__ import annotations

import numpy as np

from coloring_page.engines.chained import ChainedEngine
from coloring_page.metrics import ink_coverage


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = ChainedEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = ChainedEngine()
    result = engine.convert(synthetic_photo)

    assert np.mean(result) > 127


def test_ink_coverage_in_reasonable_band(synthetic_photo: np.ndarray) -> None:
    # Regression band, not a tuning target: catches a total-failure
    # collapse (empty edge chains, or everything redrawn as ink).
    engine = ChainedEngine()
    result = engine.convert(synthetic_photo)

    assert 0.005 <= ink_coverage(result) <= 0.40


def test_higher_thickness_adds_more_ink(synthetic_photo: np.ndarray) -> None:
    engine = ChainedEngine()
    thin = engine.convert(synthetic_photo, line_thickness=1)
    thick = engine.convert(synthetic_photo, line_thickness=3)

    assert np.sum(thick < 128) >= np.sum(thin < 128)


def test_min_path_length_affects_output(synthetic_photo: np.ndarray) -> None:
    permissive = ChainedEngine(min_path_length=5).convert(synthetic_photo)
    strict = ChainedEngine(min_path_length=100).convert(synthetic_photo)

    assert not np.array_equal(permissive, strict)


class _RecordingDebugSink:
    """Structurally satisfies the DebugSink protocol (matching .save signature)."""

    def __init__(self) -> None:
        self.stage_names: list[str] = []

    def save(self, stage_name: str, image: np.ndarray) -> None:
        self.stage_names.append(stage_name)


def test_debug_sink_receives_all_three_stages(synthetic_photo: np.ndarray) -> None:
    sink = _RecordingDebugSink()
    engine = ChainedEngine()

    engine.convert(synthetic_photo, debug=sink)

    assert sink.stage_names == ["flattened", "edge_chains", "redraw"]
