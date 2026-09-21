"""Export a ``Drawing`` to printable SVG, PDF, and PNG output.

Replaces :func:`coloring_page.drawing.rasterize`'s "stopgap" raster-only
export with real output formats, all sharing
:func:`coloring_page.page.fit_transform` for placement so an SVG, a PDF,
and a PNG of the same ``(drawing, spec)`` pair are never geometrically
inconsistent with each other.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from reportlab.lib.units import mm as MM_TO_PT
from reportlab.pdfgen import canvas as reportlab_canvas

from coloring_page.drawing import Drawing, Path
from coloring_page.page import PageSpec, fit_transform


def _path_points_mm(
    path: Path, spec: PageSpec, transform: tuple[float, float, float]
) -> np.ndarray:
    """Map a ``Path``'s normalized points to page-mm coordinates.

    Parameters
    ----------
    path : Path
        Source path, points normalized as documented in
        :mod:`coloring_page.drawing`.
    spec : PageSpec
        Unused directly here (kept for a consistent call signature across
        this module's helpers); placement is fully determined by
        ``transform``.
    transform : tuple[float, float, float]
        ``(scale, offset_x_mm, offset_y_mm)`` from
        :func:`coloring_page.page.fit_transform`.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)``, dtype ``float64``, in page-mm coordinates
        (y-down, top-left origin).
    """
    del spec
    scale, offset_x_mm, offset_y_mm = transform
    return path.points.astype(np.float64) * scale + (offset_x_mm, offset_y_mm)


def to_svg(drawing: Drawing, spec: PageSpec) -> str:
    """Render ``drawing`` to an SVG document sized for physical printing.

    One SVG user unit equals one millimeter (``viewBox="0 0 W H"`` with
    ``width="{W}mm" height="{H}mm"``), so stroke width and path
    coordinates are page-mm values directly, with no DPI assumption baked
    into the file -- it prints at the correct physical size in any
    SVG-aware consumer.

    Parameters
    ----------
    drawing : Drawing
        The vector line art to render.
    spec : PageSpec
        Target page geometry.

    Returns
    -------
    str
        A complete SVG document, including the XML declaration.
    """
    page_width_mm, page_height_mm = spec.page_size_mm
    transform = fit_transform(drawing, spec)

    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{page_width_mm}mm",
            "height": f"{page_height_mm}mm",
            "viewBox": f"0 0 {page_width_mm} {page_height_mm}",
        },
    )

    for path in drawing.paths:
        points_mm = _path_points_mm(path, spec, transform)
        commands = [f"M {points_mm[0, 0]:.3f} {points_mm[0, 1]:.3f}"]
        commands.extend(f"L {x:.3f} {y:.3f}" for x, y in points_mm[1:])
        if path.closed:
            commands.append("Z")
        ET.SubElement(
            root,
            "path",
            {
                "d": " ".join(commands),
                "fill": "none",
                "stroke": "black",
                "stroke-width": f"{spec.stroke_width_mm}",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
                "shape-rendering": "geometricPrecision",
            },
        )

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def to_pdf(drawing: Drawing, spec: PageSpec) -> bytes:
    """Render ``drawing`` to a single-page, true-vector PDF.

    Built with ``reportlab``: a BSD-licensed, pure-Python PDF writer. The
    alternative -- hand-writing PDF objects and content-stream operators
    directly against the stdlib -- was considered (the surface needed here
    is narrow: straight polyline strokes only, no text/images/fonts), but
    ``reportlab`` was chosen instead for its more robust handling of PDF
    edge cases (large path/point counts, stream chunking) without this
    module having to reimplement that itself.

    Parameters
    ----------
    drawing : Drawing
        The vector line art to render.
    spec : PageSpec
        Target page geometry.

    Returns
    -------
    bytes
        A complete single-page PDF document.
    """
    page_width_mm, page_height_mm = spec.page_size_mm
    page_width_pt = page_width_mm * MM_TO_PT
    page_height_pt = page_height_mm * MM_TO_PT
    transform = fit_transform(drawing, spec)

    buffer = io.BytesIO()
    pdf_canvas = reportlab_canvas.Canvas(buffer, pagesize=(page_width_pt, page_height_pt))
    pdf_canvas.setLineWidth(spec.stroke_width_mm * MM_TO_PT)
    pdf_canvas.setLineCap(1)  # round
    pdf_canvas.setLineJoin(1)  # round

    for path in drawing.paths:
        points_mm = _path_points_mm(path, spec, transform)
        # reportlab's canvas origin is bottom-left with a y-up axis; every
        # point produced by fit_transform is y-down (top-left origin,
        # matching the image/mm convention used throughout this project).
        # Flipping y here is what keeps the PDF right-side up instead of
        # vertically mirrored.
        pdf_path = pdf_canvas.beginPath()
        first_x, first_y = points_mm[0]
        pdf_path.moveTo(first_x * MM_TO_PT, page_height_pt - first_y * MM_TO_PT)
        for x, y in points_mm[1:]:
            pdf_path.lineTo(x * MM_TO_PT, page_height_pt - y * MM_TO_PT)
        if path.closed:
            pdf_path.close()
        pdf_canvas.drawPath(pdf_path, stroke=1, fill=0)

    pdf_canvas.showPage()
    pdf_canvas.save()
    return buffer.getvalue()


def to_png(drawing: Drawing, spec: PageSpec) -> np.ndarray:
    """Rasterize ``drawing`` onto a full page canvas at ``spec.dpi``.

    Unlike :func:`coloring_page.drawing.rasterize` (which always fills its
    canvas edge-to-edge), this renders the *whole page* -- including blank
    margins -- using the same :func:`coloring_page.page.fit_transform`
    placement as :func:`to_svg` and :func:`to_pdf`, so all three formats of
    the same ``(drawing, spec)`` agree on where content sits. This is also
    what :func:`coloring_page.validate.validate` rasterizes to measure
    output quality.

    Parameters
    ----------
    drawing : Drawing
        The vector line art to render.
    spec : PageSpec
        Target page geometry.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, shape ``(page_height_px,
        page_width_px)``, project convention: ``0`` = ink, ``255`` =
        background.
    """
    page_width_px, page_height_px = spec.page_size_px
    transform = fit_transform(drawing, spec)
    px_per_mm = spec.px_per_mm
    thickness = max(1, round(spec.stroke_width_mm * px_per_mm))

    canvas = np.full((page_height_px, page_width_px), 255, dtype=np.uint8)
    for path in drawing.paths:
        points_mm = _path_points_mm(path, spec, transform)
        points_px = (points_mm * px_per_mm).round().astype(np.int32)
        cv2.polylines(
            canvas,
            [points_px],
            isClosed=path.closed,
            color=0,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas
