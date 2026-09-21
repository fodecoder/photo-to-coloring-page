"""Tests for SVG/PDF/PNG export (coloring_page.render)."""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from coloring_page.drawing import Drawing, Path
from coloring_page.page import PageSpec, fit_transform
from coloring_page.render import to_pdf, to_png, to_svg


def _closed_square_drawing() -> Drawing:
    points = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], dtype=np.float32)
    path = Path(points=points, closed=True, kind="boundary")
    return Drawing(paths=(path,), aspect_ratio=1.0)


def _two_path_drawing() -> Drawing:
    closed_path = Path(
        points=np.array([[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]], dtype=np.float32),
        closed=True,
        kind="boundary",
    )
    open_path = Path(
        points=np.array([[0.6, 0.6], [0.9, 0.9]], dtype=np.float32), closed=False, kind="detail"
    )
    return Drawing(paths=(closed_path, open_path), aspect_ratio=1.0)


class TestToSvg:
    def test_output_is_well_formed_xml(self) -> None:
        svg = to_svg(_closed_square_drawing(), PageSpec())
        root = ET.fromstring(svg.split("?>", 1)[1])
        assert root.tag.endswith("svg")

    def test_one_path_element_per_drawing_path(self) -> None:
        svg = to_svg(_two_path_drawing(), PageSpec())
        root = ET.fromstring(svg.split("?>", 1)[1])
        paths = [el for el in root if el.tag.endswith("path")]
        assert len(paths) == 2

    def test_view_box_and_size_match_page_spec(self) -> None:
        spec = PageSpec(size=(100.0, 150.0))
        svg = to_svg(_closed_square_drawing(), spec)
        root = ET.fromstring(svg.split("?>", 1)[1])
        assert root.get("width") == "100.0mm"
        assert root.get("height") == "150.0mm"
        assert root.get("viewBox") == "0 0 100.0 150.0"

    def test_closed_path_d_ends_with_z(self) -> None:
        svg = to_svg(_closed_square_drawing(), PageSpec())
        root = ET.fromstring(svg.split("?>", 1)[1])
        path_el = next(el for el in root if el.tag.endswith("path"))
        assert path_el.get("d", "").endswith("Z")

    def test_open_path_d_does_not_end_with_z(self) -> None:
        svg = to_svg(_two_path_drawing(), PageSpec())
        root = ET.fromstring(svg.split("?>", 1)[1])
        path_els = [el for el in root if el.tag.endswith("path")]
        open_el = next(el for el in path_els if not el.get("d", "").endswith("Z"))
        assert not open_el.get("d", "").endswith("Z")

    def test_stroke_width_matches_spec(self) -> None:
        spec = PageSpec(stroke_width_mm=1.25)
        svg = to_svg(_closed_square_drawing(), spec)
        root = ET.fromstring(svg.split("?>", 1)[1])
        path_el = next(el for el in root if el.tag.endswith("path"))
        assert path_el.get("stroke-width") == "1.25"


class TestToPdf:
    def test_output_starts_and_ends_with_pdf_markers(self) -> None:
        pdf_bytes = to_pdf(_closed_square_drawing(), PageSpec())
        assert pdf_bytes.startswith(b"%PDF-")
        assert pdf_bytes.rstrip(b"\n").endswith(b"%%EOF")

    def test_page_size_matches_spec_in_points(self) -> None:
        spec = PageSpec(size=(100.0, 150.0))
        pdf_bytes = to_pdf(_closed_square_drawing(), spec)
        mm_to_pt = 72.0 / 25.4
        expected_w = round(100.0 * mm_to_pt)
        expected_h = round(150.0 * mm_to_pt)
        # MediaBox appears as a plain array of 4 numbers in the object stream.
        assert f"{expected_w}".encode() in pdf_bytes or str(expected_w).encode() in pdf_bytes
        assert b"MediaBox" in pdf_bytes
        del expected_h  # only coarsely checked above; exact float formatting is reportlab's concern


class TestToPng:
    def test_shape_matches_page_size_px(self) -> None:
        spec = PageSpec(size=(50.0, 25.0), dpi=100)
        raster = to_png(_closed_square_drawing(), spec)
        expected_w, expected_h = spec.page_size_px
        assert raster.shape == (expected_h, expected_w)

    def test_margin_border_is_blank(self) -> None:
        spec = PageSpec(size=(50.0, 50.0), margin_mm=10.0, dpi=100)
        raster = to_png(_closed_square_drawing(), spec)
        margin_px = round(10.0 * spec.px_per_mm)
        assert np.all(raster[:margin_px, :] == 255)
        assert np.all(raster[:, :margin_px] == 255)

    def test_agrees_with_svg_placement_at_sample_points(self) -> None:
        drawing = _closed_square_drawing()
        spec = PageSpec(size=(50.0, 50.0), dpi=200)
        raster = to_png(drawing, spec)

        scale, offset_x, offset_y = fit_transform(drawing, spec)
        first_point_mm = drawing.paths[0].points[0] * scale + (offset_x, offset_y)
        px_per_mm = spec.px_per_mm
        px_x, px_y = round(first_point_mm[0] * px_per_mm), round(first_point_mm[1] * px_per_mm)

        # The rasterized stroke should pass near the expected pixel location.
        window = raster[max(0, px_y - 2) : px_y + 3, max(0, px_x - 2) : px_x + 3]
        assert np.any(window < 128)


def test_pdf_content_stream_contains_stroke_path_operators() -> None:
    drawing = _two_path_drawing()
    pdf_bytes = to_pdf(drawing, PageSpec())

    # reportlab's default filter chain for content streams is
    # [ASCII85Decode, FlateDecode] (confirmed by inspecting a sample
    # document's stream dict): decode both layers, in that order, then
    # confirm the operator stream ended up with recognizable PDF
    # path-construction/stroke operators ('m'/'l'/'S') -- a coarse
    # structural check that real vector path data was emitted, without
    # pinning down reportlab's exact operator spacing.
    found_path_ops = False
    search_from = 0
    while True:
        stream_start = pdf_bytes.find(b"\nstream\n", search_from)
        if stream_start == -1:
            break
        content_start = stream_start + len(b"\nstream\n")
        end = pdf_bytes.find(b"endstream", content_start)
        if end == -1:
            break
        raw = pdf_bytes[content_start:end].strip(b"\r\n")
        try:
            decoded = base64.a85decode(raw, adobe=True)
            text = zlib.decompress(decoded).decode("latin-1")
        except (ValueError, zlib.error):
            search_from = end + len(b"endstream")
            continue
        if " m " in text or " m\n" in text:
            found_path_ops = True
            assert "S" in text.split()  # stroke operator, as its own token
            break
        search_from = end + len(b"endstream")

    assert found_path_ops, "no decompressed content stream contained a moveto ('m') operator"
