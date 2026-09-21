"""Tests for the flatten-chain-redraw (chained) conversion engine."""

from __future__ import annotations

import numpy as np

from coloring_page.drawing import rasterize
from coloring_page.engines.chained import ChainedEngine
from coloring_page.validate import ink_coverage


def test_requires_serial_execution_is_false() -> None:
    assert ChainedEngine.requires_serial_execution is False


def test_output_is_a_nonempty_drawing(synthetic_photo: np.ndarray) -> None:
    engine = ChainedEngine()
    drawing = engine.convert(synthetic_photo)

    assert len(drawing.paths) > 0
    assert drawing.aspect_ratio == synthetic_photo.shape[1] / synthetic_photo.shape[0]


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = ChainedEngine()
    drawing = engine.convert(synthetic_photo)
    result = rasterize(drawing, long_side_px=max(synthetic_photo.shape[:2]))

    assert np.mean(result) > 127


def test_ink_coverage_in_reasonable_band(synthetic_photo: np.ndarray) -> None:
    # Regression band, not a tuning target: catches a total-failure
    # collapse (empty edge chains, or everything redrawn as ink).
    engine = ChainedEngine()
    drawing = engine.convert(synthetic_photo)
    result = rasterize(drawing, long_side_px=max(synthetic_photo.shape[:2]))

    assert 0.005 <= ink_coverage(result) <= 0.40


def test_higher_thickness_adds_more_ink(synthetic_photo: np.ndarray) -> None:
    engine = ChainedEngine()
    drawing = engine.convert(synthetic_photo)
    long_side = max(synthetic_photo.shape[:2])
    thin = rasterize(drawing, long_side_px=long_side, line_thickness=1)
    thick = rasterize(drawing, long_side_px=long_side, line_thickness=3)

    assert np.sum(thick < 128) >= np.sum(thin < 128)


def test_min_path_length_affects_output(synthetic_photo: np.ndarray) -> None:
    permissive = ChainedEngine(min_path_length=5).convert(synthetic_photo)
    strict = ChainedEngine(min_path_length=100).convert(synthetic_photo)

    assert len(permissive.paths) != len(strict.paths)


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
