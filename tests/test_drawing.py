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
from coloring_page.page import PageSpec
from coloring_page.validate import validate


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

    def test_plus_junction_continues_straight_through(self) -> None:
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[30, 5:55] = 255  # horizontal bar
        mask[5:55, 30] = 255  # vertical bar

        paths = paths_from_mask(mask)

        # Opposite arms are joined through the junction: one horizontal
        # and one vertical stroke, each spanning (almost) the full bar,
        # rather than four arms split at the crossing.
        assert len(paths) == 2
        assert all(not p.closed for p in paths)
        extents = sorted(tuple(np.ptp(p.points * 60, axis=0).round()) for p in paths)
        assert extents[0][0] <= 2 and extents[0][1] >= 45
        assert extents[1][0] >= 45 and extents[1][1] <= 2

    def test_t_junction_keeps_bar_whole_and_stem_separate(self) -> None:
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[10, 5:55] = 255  # bar
        mask[10:55, 30] = 255  # stem

        paths = paths_from_mask(mask)

        assert len(paths) == 2
        lengths_px = sorted(p.length * 60 for p in paths)
        assert lengths_px[0] == pytest.approx(44, abs=3)  # stem
        assert lengths_px[1] == pytest.approx(49, abs=3)  # bar, unbroken

    @pytest.mark.parametrize("thickness", [1, 3, 5])
    def test_circle_traces_one_closed_path(self, thickness: int) -> None:
        # Thickness > 1 is the regression case: Zhang-Suen leaves
        # staircase corners along every curve of a thick stroke, which
        # used to be classified as junctions and shatter the circle into
        # dozens of short open fragments.
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.circle(mask, (60, 60), 40, 255, thickness=thickness)

        paths = paths_from_mask(mask)

        assert len(paths) == 1
        assert paths[0].closed is True
        assert len(paths[0].points) >= 8
        assert paths[0].length * 120 == pytest.approx(2 * np.pi * 40, rel=0.1)

    @pytest.mark.parametrize("thickness", [1, 3, 5])
    def test_acute_parallelogram_traces_one_closed_path(self, thickness: int) -> None:
        # A rectangle sheared to a 30-degree corner: thinning a thick
        # stroke grows a spur into an acute corner, which must be pruned
        # rather than turning the outline into an open 3-way junction.
        corners = np.array([[20, 170], [170, 170], [300, 95], [150, 95]], dtype=np.int32)
        mask = np.zeros((200, 320), dtype=np.uint8)
        cv2.polylines(mask, [corners], isClosed=True, color=255, thickness=thickness)

        paths = paths_from_mask(mask)

        assert len(paths) == 1
        assert paths[0].closed is True
        perimeter = 2 * (150 + float(np.hypot(130, 75)))
        assert paths[0].length * 320 == pytest.approx(perimeter, rel=0.1)

    @pytest.mark.parametrize("thickness", [1, 3])
    def test_open_curve_is_one_path_without_collinear_interior_points(self, thickness: int) -> None:
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(mask, (100, 100), (60, 40), 0, 0, 200, 255, thickness=thickness)

        paths = paths_from_mask(mask)

        assert len(paths) == 1
        assert paths[0].closed is False
        pixels = np.round(paths[0].points * 200).astype(np.int64)
        incoming = pixels[1:-1] - pixels[:-2]
        outgoing = pixels[2:] - pixels[1:-1]
        cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        # Every interior vertex is a real turn: no point lies exactly on
        # the segment between its neighbors (neither per-pixel run points
        # nor join points left behind by merging edges).
        assert np.all(cross != 0)

    def test_no_path_has_fewer_than_two_points(self) -> None:
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[30, 5:55] = 255
        mask[5:55, 30] = 255
        cv2.circle(mask, (10, 10), 3, 255, thickness=1)

        for path in paths_from_mask(mask):
            assert len(path.points) >= 2

    def test_three_closed_shapes_have_no_dangling_endpoints(self) -> None:
        # Regression: validate() counts 2 dangling endpoints per open
        # path, so every closed outline split at a (real or staircase)
        # junction used to show up here.
        mask = _three_closed_shapes_mask()

        paths = paths_from_mask(mask)
        drawing = Drawing(paths=paths, aspect_ratio=mask.shape[1] / mask.shape[0])
        report = validate(drawing, PageSpec())

        assert len(paths) == 3
        assert all(p.closed for p in paths)
        assert report.dangling_endpoints == 0


def _three_closed_shapes_mask() -> np.ndarray:
    """A circle, an acute parallelogram, and a triangle, 3px strokes, well separated."""
    mask = np.zeros((240, 480), dtype=np.uint8)
    cv2.circle(mask, (80, 120), 55, 255, thickness=3)
    parallelogram = np.array([[170, 180], [290, 180], [350, 60], [230, 60]], dtype=np.int32)
    cv2.polylines(mask, [parallelogram], isClosed=True, color=255, thickness=3)
    triangle = np.array([[380, 190], [460, 190], [420, 40]], dtype=np.int32)
    cv2.polylines(mask, [triangle], isClosed=True, color=255, thickness=3)
    return mask


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
