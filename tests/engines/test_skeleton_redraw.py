"""Tests for the flatten-skeletonize-redraw (skeleton) conversion engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from coloring_page.drawing import rasterize
from coloring_page.engines.skeleton_redraw import SkeletonRedrawEngine
from coloring_page.validate import ink_coverage


def test_requires_serial_execution_is_false() -> None:
    assert SkeletonRedrawEngine.requires_serial_execution is False


def test_output_is_a_nonempty_drawing(synthetic_photo: np.ndarray) -> None:
    engine = SkeletonRedrawEngine()
    drawing = engine.convert(synthetic_photo)

    assert len(drawing.paths) > 0
    assert drawing.aspect_ratio == synthetic_photo.shape[1] / synthetic_photo.shape[0]


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = SkeletonRedrawEngine()
    drawing = engine.convert(synthetic_photo)
    result = rasterize(drawing, long_side_px=max(synthetic_photo.shape[:2]))

    assert np.mean(result) > 127


def test_ink_coverage_in_reasonable_band(synthetic_photo: np.ndarray) -> None:
    # Regression band, not a tuning target: catches a total-failure
    # collapse (empty skeleton, or everything redrawn as ink).
    engine = SkeletonRedrawEngine()
    drawing = engine.convert(synthetic_photo)
    result = rasterize(drawing, long_side_px=max(synthetic_photo.shape[:2]))

    assert 0.005 <= ink_coverage(result) <= 0.40


def test_higher_thickness_adds_more_ink(synthetic_photo: np.ndarray) -> None:
    engine = SkeletonRedrawEngine()
    drawing = engine.convert(synthetic_photo)
    long_side = max(synthetic_photo.shape[:2])
    thin = rasterize(drawing, long_side_px=long_side, line_thickness=1)
    thick = rasterize(drawing, long_side_px=long_side, line_thickness=3)

    assert np.sum(thick < 128) >= np.sum(thin < 128)


class _RecordingDebugSink:
    """Structurally satisfies the DebugSink protocol (matching .save signature)."""

    def __init__(self) -> None:
        self.stage_names: list[str] = []

    def save(self, stage_name: str, image: np.ndarray) -> None:
        self.stage_names.append(stage_name)


def test_debug_sink_receives_all_stages(synthetic_photo: np.ndarray) -> None:
    sink = _RecordingDebugSink()
    engine = SkeletonRedrawEngine()

    engine.convert(synthetic_photo, debug=sink)

    # No separate "skeleton" stage anymore: thinning/branch-tracing now
    # happens inside paths_from_mask, which has no raster intermediate to
    # report -- "redraw" is the vector Drawing's own rasterized preview.
    assert sink.stage_names == ["flattened", "edges", "redraw"]


def _horizontal_gradient(height: int, width: int) -> np.ndarray:
    ramp = np.tile(np.linspace(60, 200, width).astype(np.uint8), (height, 1))
    return np.dstack([ramp, ramp, ramp])


def test_uniform_image_produces_no_paths() -> None:
    image = np.full((200, 300, 3), 180, dtype=np.uint8)

    drawing = SkeletonRedrawEngine().convert(image)

    assert len(drawing.paths) == 0


@pytest.mark.parametrize("background", ["uniform", "horizontal_gradient"])
def test_centered_circle_has_no_path_near_border(background: str) -> None:
    # The gradient background is the real regression case: l0Smooth's
    # periodic boundary treats the dark left edge as adjacent to the
    # light right edge, which used to produce a step along the whole
    # border that got traced as a frame. A uniform background has equal
    # opposite edges, so it can't trigger the artifact on its own.
    height, width = 200, 300
    if background == "uniform":
        image = np.full((height, width, 3), 230, dtype=np.uint8)
    else:
        image = _horizontal_gradient(height, width)
    cv2.circle(image, (width // 2, height // 2), 50, (20, 20, 20), thickness=-1)

    drawing = SkeletonRedrawEngine().convert(image)

    assert len(drawing.paths) >= 1
    long_side = max(height, width)
    for path in drawing.paths:
        x, y = (path.points * long_side).T
        near_border = (x <= 2) | (y <= 2) | (x >= width - 3) | (y >= height - 3)
        assert not np.any(near_border), "a path runs within 2px of the image border"
