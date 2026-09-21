"""Tests for colorability validation (coloring_page.validate).

Uses synthetic figures with known ground truth (a circle, a circle with a
small gap, two rectangles) -- exactly the kind of test that would have
caught the "nothing checks whether contours are closed" gap this module
fixes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from coloring_page.drawing import Drawing, Path
from coloring_page.page import PageSpec, fit_transform
from coloring_page.validate import QualityReport, ink_coverage, validate

#: Small page/resolution used throughout so tests run fast; large enough
#: that a 5px gap and small stroke widths remain meaningful.
_SPEC = PageSpec(size=(50.0, 50.0), margin_mm=5.0, dpi=100)


def _circle_points(
    *,
    center: tuple[float, float] = (0.5, 0.5),
    radius: float = 0.4,
    start: float = 0.0,
    end: float = 2 * math.pi,
    n: int = 128,
) -> np.ndarray:
    angles = np.linspace(start, end, n, dtype=np.float64)
    cx, cy = center
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _closed_circle_drawing(radius: float = 0.4) -> Drawing:
    path = Path(points=_circle_points(radius=radius), closed=True, kind="boundary")
    return Drawing(paths=(path,), aspect_ratio=1.0)


def _gapped_circle_drawing(radius: float = 0.4, *, gap_px: float = 5.0) -> Drawing:
    """A circle traced as one open Path with a gap of roughly ``gap_px`` pixels."""
    scale, _, _ = fit_transform(_closed_circle_drawing(radius), _SPEC)
    px_per_unit = scale * _SPEC.px_per_mm
    radius_px = radius * px_per_unit
    gap_angle = gap_px / radius_px

    points = _circle_points(radius=radius, start=gap_angle / 2, end=2 * math.pi - gap_angle / 2)
    path = Path(points=points, closed=False, kind="boundary")
    return Drawing(paths=(path,), aspect_ratio=1.0)


def _rectangle_points(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


class TestInkCoverage:
    def test_all_white_is_zero(self) -> None:
        blank = np.full((10, 10), 255, dtype=np.uint8)
        assert ink_coverage(blank) == 0.0

    def test_all_black_is_one(self) -> None:
        solid = np.full((10, 10), 0, dtype=np.uint8)
        assert ink_coverage(solid) == 1.0

    def test_threshold_boundary_at_128(self) -> None:
        image = np.full((1, 2), 128, dtype=np.uint8)
        image[0, 0] = 127
        assert ink_coverage(image) == pytest.approx(0.5)


class TestEnclosedRegions:
    def test_closed_circle_has_one_enclosed_region_and_no_leaks(self) -> None:
        report = validate(_closed_circle_drawing(), _SPEC)
        assert report.enclosed_regions == 1
        assert report.leaking_regions == 0


class TestLeakingRegions:
    def test_circle_with_small_gap_leaks(self) -> None:
        report = validate(_gapped_circle_drawing(gap_px=5.0), _SPEC)
        assert report.leaking_regions > 0
        # The interior is flood-reached, not enclosed, in the real (ungapped) raster.
        assert report.enclosed_regions == 0


class TestEnclosedRegionsMultiple:
    def test_two_separate_rectangles(self) -> None:
        rect_a = Path(points=_rectangle_points(0.1, 0.1, 0.3, 0.3), closed=True, kind="boundary")
        rect_b = Path(points=_rectangle_points(0.5, 0.5, 0.8, 0.8), closed=True, kind="boundary")
        drawing = Drawing(paths=(rect_a, rect_b), aspect_ratio=1.0)

        report = validate(drawing, _SPEC)
        assert report.enclosed_regions == 2

    def test_two_edge_sharing_rectangles(self) -> None:
        # A shared ink edge separates the interiors -- it must not merge
        # them into a single enclosed region.
        rect_a = Path(points=_rectangle_points(0.1, 0.1, 0.45, 0.8), closed=True, kind="boundary")
        rect_b = Path(points=_rectangle_points(0.45, 0.1, 0.8, 0.8), closed=True, kind="boundary")
        drawing = Drawing(paths=(rect_a, rect_b), aspect_ratio=1.0)

        report = validate(drawing, _SPEC)
        assert report.enclosed_regions == 2


class TestDanglingEndpoints:
    def test_open_path_has_two_dangling_endpoints(self) -> None:
        path = Path(
            points=np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32), closed=False, kind="detail"
        )
        drawing = Drawing(paths=(path,), aspect_ratio=1.0)
        report = validate(drawing, _SPEC)
        assert report.dangling_endpoints == 2

    def test_all_closed_paths_have_no_dangling_endpoints(self) -> None:
        report = validate(_closed_circle_drawing(), _SPEC)
        assert report.dangling_endpoints == 0


class TestRegionArea:
    def test_matches_analytic_circle_area_within_tolerance(self) -> None:
        radius = 0.4
        report = validate(_closed_circle_drawing(radius), _SPEC)

        scale, _, _ = fit_transform(_closed_circle_drawing(radius), _SPEC)
        radius_mm = radius * scale
        expected_area_mm2 = math.pi * radius_mm**2

        assert report.min_region_area_mm2 is not None
        assert report.min_region_area_mm2 == pytest.approx(expected_area_mm2, rel=0.15)


class TestQualityReportPassed:
    def _base_kwargs(self) -> dict[str, object]:
        return {
            "ink_coverage": 0.05,
            "enclosed_regions": 1,
            "leaking_regions": 0,
            "min_region_area_mm2": 10.0,
            "region_area_mm2_p5": 10.0,
            "dangling_endpoints": 0,
        }

    def test_passes_when_everything_is_in_band(self) -> None:
        assert QualityReport(**self._base_kwargs()).passed is True

    def test_fails_when_ink_coverage_out_of_band(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["ink_coverage"] = 0.5
        assert QualityReport(**kwargs).passed is False

    def test_fails_when_leaking_regions_exceed_max(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["leaking_regions"] = 1
        assert QualityReport(**kwargs).passed is False

    def test_fails_when_dangling_endpoints_exceed_max(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["dangling_endpoints"] = 2
        assert QualityReport(**kwargs).passed is False

    def test_fails_when_min_region_area_below_floor(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["min_region_area_mm2"] = 0.1
        assert QualityReport(**kwargs).passed is False

    def test_passes_when_no_enclosed_regions_area_is_vacuously_ok(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["min_region_area_mm2"] = None
        kwargs["region_area_mm2_p5"] = None
        assert QualityReport(**kwargs).passed is True
