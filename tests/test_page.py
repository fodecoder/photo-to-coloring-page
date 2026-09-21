"""Tests for print-page geometry (PageSpec) and content placement (fit_transform)."""

from __future__ import annotations

import numpy as np
import pytest

from coloring_page.drawing import Drawing, Path
from coloring_page.page import PageSpec, fit_transform


def _make_drawing(aspect_ratio: float) -> Drawing:
    path = Path(
        points=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), closed=False, kind="boundary"
    )
    return Drawing(paths=(path,), aspect_ratio=aspect_ratio)


class TestPageSizeMm:
    def test_a4_portrait(self) -> None:
        spec = PageSpec(size="A4", orientation="portrait")
        assert spec.page_size_mm == (210.0, 297.0)

    def test_a4_landscape_swaps_dimensions(self) -> None:
        spec = PageSpec(size="A4", orientation="landscape")
        assert spec.page_size_mm == (297.0, 210.0)

    def test_a5_portrait(self) -> None:
        spec = PageSpec(size="A5")
        assert spec.page_size_mm == (148.0, 210.0)

    def test_letter_portrait(self) -> None:
        spec = PageSpec(size="LETTER")
        assert spec.page_size_mm == (215.9, 279.4)

    def test_explicit_tuple_passes_through_unchanged(self) -> None:
        spec = PageSpec(size=(100.0, 50.0), orientation="landscape")
        assert spec.page_size_mm == (100.0, 50.0)

    def test_explicit_tuple_rejects_non_positive_values(self) -> None:
        spec = PageSpec(size=(0.0, 50.0))
        with pytest.raises(ValueError, match="positive"):
            _ = spec.page_size_mm

    def test_unknown_named_size_raises(self) -> None:
        spec = PageSpec(size="B4")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown page size"):
            _ = spec.page_size_mm


class TestUsableSizeMm:
    def test_subtracts_margins_from_both_axes(self) -> None:
        spec = PageSpec(size=(100.0, 200.0), margin_mm=10.0)
        assert spec.usable_size_mm == (80.0, 180.0)

    def test_raises_when_margins_exceed_half_the_page(self) -> None:
        spec = PageSpec(size=(20.0, 20.0), margin_mm=15.0)
        with pytest.raises(ValueError, match="no usable area"):
            _ = spec.usable_size_mm


class TestPixelConversions:
    def test_px_per_mm(self) -> None:
        spec = PageSpec(dpi=254)  # 254/25.4 = 10 px/mm exactly
        assert spec.px_per_mm == pytest.approx(10.0)

    def test_page_size_px(self) -> None:
        spec = PageSpec(size=(25.4, 50.8), dpi=100)
        assert spec.page_size_px == (100, 200)


class TestFitTransform:
    def test_letterboxes_wide_content_in_a_taller_usable_area(self) -> None:
        # A very wide drawing (aspect_ratio=4) in a page whose usable area
        # is nearly square -- width is the binding constraint.
        drawing = _make_drawing(aspect_ratio=4.0)
        spec = PageSpec(size=(110.0, 110.0), margin_mm=10.0)  # usable: 90x90

        scale, offset_x, offset_y = fit_transform(drawing, spec)

        # content is 1.0 x 0.25 units; scale must be min(90/1.0, 90/0.25) = 90.
        assert scale == pytest.approx(90.0)
        assert offset_x == pytest.approx(10.0)  # fills the usable width exactly
        # content height = 0.25 * 90 = 22.5mm, centered in 90mm usable height.
        assert offset_y == pytest.approx(10.0 + (90.0 - 22.5) / 2)

    def test_letterboxes_tall_content_in_a_wider_usable_area(self) -> None:
        drawing = _make_drawing(aspect_ratio=0.25)
        spec = PageSpec(size=(110.0, 110.0), margin_mm=10.0)

        scale, offset_x, offset_y = fit_transform(drawing, spec)

        assert scale == pytest.approx(90.0)
        assert offset_y == pytest.approx(10.0)
        assert offset_x == pytest.approx(10.0 + (90.0 - 22.5) / 2)

    def test_square_content_in_square_usable_area_has_no_offset_needed(self) -> None:
        drawing = _make_drawing(aspect_ratio=1.0)
        spec = PageSpec(size=(120.0, 120.0), margin_mm=10.0)

        scale, offset_x, offset_y = fit_transform(drawing, spec)

        assert scale == pytest.approx(100.0)
        assert offset_x == pytest.approx(10.0)
        assert offset_y == pytest.approx(10.0)
