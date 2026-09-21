"""Tests for the vector Drawing/Path domain model."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from coloring_page.drawing import (
    Drawing,
    Path,
    paths_from_mask,
    paths_from_point_chains,
    rasterize,
)


def _make_path(points: list[tuple[float, float]], *, closed: bool = False) -> Path:
    return Path(points=np.array(points, dtype=np.float32), closed=closed, kind="boundary")


class TestPathLength:
    def test_open_path_sums_segment_lengths(self) -> None:
        path = _make_path([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)])
        assert path.length == pytest.approx(3.0 + 4.0)

    def test_closed_path_includes_closing_segment(self) -> None:
        path = _make_path([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], closed=True)
        assert path.length == pytest.approx(4.0)

    def test_open_path_excludes_closing_segment(self) -> None:
        path = _make_path([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], closed=False)
        assert path.length == pytest.approx(3.0)


class TestDrawingFilter:
    def test_keeps_only_matching_paths(self) -> None:
        short = _make_path([(0.0, 0.0), (0.1, 0.0)])
        long = _make_path([(0.0, 0.0), (0.9, 0.0)])
        drawing = Drawing(paths=(short, long), aspect_ratio=1.0)

        filtered = drawing.filter(lambda p: p.length >= 0.5)

        assert filtered.paths == (long,)
        assert filtered.aspect_ratio == drawing.aspect_ratio

    def test_returns_new_instance(self) -> None:
        path = _make_path([(0.0, 0.0), (1.0, 0.0)])
        drawing = Drawing(paths=(path,), aspect_ratio=1.0)
        filtered = drawing.filter(lambda p: True)
        assert filtered is not drawing


class TestDrawingSimplify:
    def test_simplifies_collinear_points(self) -> None:
        # Three collinear points simplify to two under a nonzero epsilon.
        path = _make_path([(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)])
        drawing = Drawing(paths=(path,), aspect_ratio=1.0)

        simplified = drawing.simplify(epsilon=0.01)

        assert len(simplified.paths[0].points) == 2


class TestDrawingBounds:
    def test_computes_bounding_box_across_paths(self) -> None:
        a = _make_path([(0.1, 0.2), (0.3, 0.4)])
        b = _make_path([(0.5, 0.05), (0.6, 0.9)])
        drawing = Drawing(paths=(a, b), aspect_ratio=1.0)

        assert drawing.bounds() == pytest.approx((0.1, 0.05, 0.6, 0.9))

    def test_raises_on_empty_drawing(self) -> None:
        drawing = Drawing(paths=(), aspect_ratio=1.0)
        with pytest.raises(ValueError, match="no paths"):
            drawing.bounds()


class TestPathsFromMask:
    def test_empty_mask_returns_no_paths(self) -> None:
        mask = np.zeros((50, 50), dtype=np.uint8)
        assert paths_from_mask(mask) == ()

    def test_straight_line_traces_one_open_path(self) -> None:
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, 10:40] = 255

        paths = paths_from_mask(mask)

        assert len(paths) == 1
        assert paths[0].closed is False
        assert len(paths[0].points) >= 2
        assert np.all((paths[0].points >= 0.0) & (paths[0].points <= 1.0))

    def test_plus_junction_traces_four_branches(self) -> None:
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[30, 5:55] = 255  # horizontal bar
        mask[5:55, 30] = 255  # vertical bar

        paths = paths_from_mask(mask)

        # Four arms radiating from the center junction, none closed, and
        # every branch traced (no silently dropped arm).
        assert len(paths) == 4
        assert all(not p.closed for p in paths)
        assert all(len(p.points) >= 2 for p in paths)

    def test_circle_traces_one_closed_path(self) -> None:
        mask = np.zeros((80, 80), dtype=np.uint8)
        cv2.circle(mask, (40, 40), 25, 255, thickness=1)

        paths = paths_from_mask(mask)

        assert len(paths) == 1
        assert paths[0].closed is True
        assert len(paths[0].points) >= 8

    def test_no_path_has_fewer_than_two_points(self) -> None:
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[30, 5:55] = 255
        mask[5:55, 30] = 255
        cv2.circle(mask, (10, 10), 3, 255, thickness=1)

        for path in paths_from_mask(mask):
            assert len(path.points) >= 2


class TestPathsFromPointChains:
    def test_normalizes_and_wraps_chains(self) -> None:
        chain = np.array([[0, 0], [50, 0], [50, 100]], dtype=np.int32)

        paths = paths_from_point_chains([chain], (100, 100), closed=False)

        assert len(paths) == 1
        np.testing.assert_allclose(paths[0].points, [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]], atol=1e-6)

    def test_drops_chains_with_fewer_than_two_points(self) -> None:
        chain = np.array([[5, 5]], dtype=np.int32)
        assert paths_from_point_chains([chain], (100, 100)) == ()


class TestRasterize:
    def test_wide_aspect_ratio_produces_wider_canvas(self) -> None:
        drawing = Drawing(paths=(_make_path([(0.0, 0.5), (1.0, 0.5)]),), aspect_ratio=2.0)
        canvas = rasterize(drawing, long_side_px=100)
        assert canvas.shape == (50, 100)

    def test_tall_aspect_ratio_produces_taller_canvas(self) -> None:
        drawing = Drawing(paths=(_make_path([(0.5, 0.0), (0.5, 1.0)]),), aspect_ratio=0.5)
        canvas = rasterize(drawing, long_side_px=100)
        assert canvas.shape == (100, 50)

    def test_ink_lands_near_expected_pixels(self) -> None:
        drawing = Drawing(paths=(_make_path([(0.0, 0.5), (1.0, 0.5)]),), aspect_ratio=1.0)
        canvas = rasterize(drawing, long_side_px=100)
        assert np.any(canvas[48:52, :] < 128)
        assert np.all(canvas[:10, :] == 255)

    def test_round_trips_a_closed_square(self) -> None:
        square = _make_path([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], closed=True)
        drawing = Drawing(paths=(square,), aspect_ratio=1.0)
        canvas = rasterize(drawing, long_side_px=100)
        # All four edges should have ink.
        assert np.any(canvas[10, :] < 128)
        assert np.any(canvas[90, :] < 128)
        assert np.any(canvas[:, 10] < 128)
        assert np.any(canvas[:, 90] < 128)
