"""Export an ``Artwork`` to printable SVG, PDF, and PNG output.

Replaces :func:`coloring_page.drawing.rasterize`'s "stopgap" raster-only
export with real output formats, all sharing
:func:`coloring_page.page.fit_transform` for placement so an SVG, a PDF,
and a PNG of the same ``(artwork, spec)`` pair are never geometrically
inconsistent with each other.

Each function dispatches on whether it was given a
:class:`~coloring_page.drawing.Drawing` or a
:class:`~coloring_page.artwork.RasterArtwork` -- see
:mod:`coloring_page.artwork` for why both exist. The two branches share
placement (:func:`~coloring_page.page.fit_transform`) but not much else:
a ``Drawing`` is stroked as vector paths, while a ``RasterArtwork`` is
resampled and embedded as a pixel image, so unifying the two bodies would
only obscure that they're fundamentally different operations.
"""

from __future__ import annotations

import base64
import io
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from reportlab.lib.units import mm as MM_TO_PT
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as reportlab_canvas

from coloring_page.artwork import Artwork, RasterArtwork
from coloring_page.drawing import Path
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


def _content_size_mm(artwork: Artwork, spec: PageSpec) -> tuple[float, float, float, float, float]:
    """Shared placement math: scale plus content/offset in mm, from ``fit_transform``.

    Every ``RasterArtwork`` branch below needs the content's mm size (to
    pick a resize target), not just the point-mapping ``transform`` a
    ``Drawing`` branch uses directly -- this factors that shared
    reconstruction out of :func:`fit_transform`'s return value.

    Returns
    -------
    tuple[float, float, float, float, float]
        ``(scale, offset_x_mm, offset_y_mm, content_width_mm, content_height_mm)``.
    """
    scale, offset_x_mm, offset_y_mm = fit_transform(artwork, spec)
    if artwork.aspect_ratio >= 1:
        content_width_units, content_height_units = 1.0, 1.0 / artwork.aspect_ratio
    else:
        content_width_units, content_height_units = artwork.aspect_ratio, 1.0
    return (
        scale,
        offset_x_mm,
        offset_y_mm,
        content_width_units * scale,
        content_height_units * scale,
    )


def _encode_png(image: np.ndarray) -> bytes:
    """Lossless PNG-encode a single-channel ``uint8`` image."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to PNG-encode raster artwork.")
    return bytes(buffer)


def _resample_raster(image: np.ndarray, target_width_px: int, target_height_px: int) -> np.ndarray:
    """Resize a raster image to a target pixel box, never with nearest-neighbor.

    Nearest-neighbor is the one interpolation mode that would throw away
    :class:`~coloring_page.artwork.RasterArtwork`'s antialiasing in this
    last resampling step -- exactly the thing the raster representation
    exists to preserve. ``INTER_AREA`` is used when downscaling (its
    area-averaging is the standard choice for shrinking without aliasing),
    ``INTER_LANCZOS4`` when upscaling (sharper than bilinear/bicubic for
    enlarging fine line art).

    Parameters
    ----------
    image : np.ndarray
    target_width_px : int
    target_height_px : int

    Returns
    -------
    np.ndarray
    """
    src_h, src_w = image.shape[:2]
    downscaling = target_width_px * target_height_px <= src_w * src_h
    interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_LANCZOS4
    return cv2.resize(image, (target_width_px, target_height_px), interpolation=interpolation)


def to_svg(artwork: Artwork, spec: PageSpec) -> str:
    """Render ``artwork`` to an SVG document sized for physical printing.

    One SVG user unit equals one millimeter (``viewBox="0 0 W H"`` with
    ``width="{W}mm" height="{H}mm"``), so coordinates are page-mm values
    directly, with no DPI assumption baked into the file.

    For a :class:`~coloring_page.drawing.Drawing`, this prints at the
    correct physical size in any SVG-aware consumer and stays fully
    resolution-independent. For a
    :class:`~coloring_page.artwork.RasterArtwork`, the SVG is a
    *container*, not a vector drawing -- it embeds a single base64 PNG
    ``<image>`` element sized to :attr:`~coloring_page.page.PageSpec.dpi`,
    it is not resolution-independent the way the ``Drawing`` branch is, and
    it must never be re-vectorized to "make it a real SVG": that is exactly
    the transformation :mod:`coloring_page.artwork` exists to avoid.

    Parameters
    ----------
    artwork : Artwork
        The line art to render.
    spec : PageSpec
        Target page geometry.

    Returns
    -------
    str
        A complete SVG document, including the XML declaration.
    """
    page_width_mm, page_height_mm = spec.page_size_mm

    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{page_width_mm}mm",
            "height": f"{page_height_mm}mm",
            "viewBox": f"0 0 {page_width_mm} {page_height_mm}",
        },
    )

    if isinstance(artwork, RasterArtwork):
        _, offset_x_mm, offset_y_mm, content_width_mm, content_height_mm = _content_size_mm(
            artwork, spec
        )
        px_per_mm = spec.px_per_mm
        target_w = max(1, round(content_width_mm * px_per_mm))
        target_h = max(1, round(content_height_mm * px_per_mm))
        resized = _resample_raster(artwork.image, target_w, target_h)
        png_b64 = base64.b64encode(_encode_png(resized)).decode("ascii")
        ET.SubElement(
            root,
            "image",
            {
                "x": f"{offset_x_mm:.3f}",
                "y": f"{offset_y_mm:.3f}",
                "width": f"{content_width_mm:.3f}",
                "height": f"{content_height_mm:.3f}",
                "href": f"data:image/png;base64,{png_b64}",
                "preserveAspectRatio": "none",
            },
        )
    else:
        transform = fit_transform(artwork, spec)
        for path in artwork.paths:
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


def to_pdf(artwork: Artwork, spec: PageSpec) -> bytes:
    """Render ``artwork`` to a single-page PDF.

    Built with ``reportlab``: a BSD-licensed, pure-Python PDF writer. The
    alternative -- hand-writing PDF objects and content-stream operators
    directly against the stdlib -- was considered (the surface needed here
    is narrow: straight polyline strokes and a single embedded image, no
    text/fonts), but ``reportlab`` was chosen instead for its more robust
    handling of PDF edge cases (large path/point counts, stream chunking,
    image XObjects) without this module having to reimplement that itself.

    For a :class:`~coloring_page.drawing.Drawing`, this is a true-vector
    PDF. For a :class:`~coloring_page.artwork.RasterArtwork`, the image is
    embedded at >= :attr:`~coloring_page.page.PageSpec.dpi` via a lossless
    PNG round-trip (no JPEG recompression, which would reintroduce
    blocking artifacts on top of already-subtle antialiasing).

    Parameters
    ----------
    artwork : Artwork
        The line art to render.
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

    buffer = io.BytesIO()
    pdf_canvas = reportlab_canvas.Canvas(buffer, pagesize=(page_width_pt, page_height_pt))

    if isinstance(artwork, RasterArtwork):
        _, offset_x_mm, offset_y_mm, content_width_mm, content_height_mm = _content_size_mm(
            artwork, spec
        )
        px_per_mm = spec.px_per_mm
        target_w = max(1, round(content_width_mm * px_per_mm))
        target_h = max(1, round(content_height_mm * px_per_mm))
        resized = _resample_raster(artwork.image, target_w, target_h)
        reader = ImageReader(io.BytesIO(_encode_png(resized)))
        # reportlab's canvas origin is bottom-left with a y-up axis; every
        # offset produced by fit_transform is y-down (top-left origin,
        # matching the image/mm convention used throughout this project).
        pdf_canvas.drawImage(
            reader,
            offset_x_mm * MM_TO_PT,
            page_height_pt - (offset_y_mm + content_height_mm) * MM_TO_PT,
            width=content_width_mm * MM_TO_PT,
            height=content_height_mm * MM_TO_PT,
        )
    else:
        transform = fit_transform(artwork, spec)
        pdf_canvas.setLineWidth(spec.stroke_width_mm * MM_TO_PT)
        pdf_canvas.setLineCap(1)  # round
        pdf_canvas.setLineJoin(1)  # round
        for path in artwork.paths:
            points_mm = _path_points_mm(path, spec, transform)
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


def to_png(artwork: Artwork, spec: PageSpec) -> np.ndarray:
    """Rasterize ``artwork`` onto a full page canvas at ``spec.dpi``.

    Unlike :func:`coloring_page.drawing.rasterize` (which always fills its
    canvas edge-to-edge), this renders the *whole page* -- including blank
    margins -- using the same :func:`coloring_page.page.fit_transform`
    placement as :func:`to_svg` and :func:`to_pdf`, so all three formats of
    the same ``(artwork, spec)`` agree on where content sits. This is also
    what :func:`coloring_page.validate.validate` rasterizes to measure
    output quality.

    A :class:`~coloring_page.artwork.RasterArtwork` is resampled into the
    content box with :func:`_resample_raster` (never nearest-neighbor, to
    keep its antialiasing) and pasted onto the page canvas.

    Parameters
    ----------
    artwork : Artwork
        The line art to render.
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
    canvas = np.full((page_height_px, page_width_px), 255, dtype=np.uint8)

    if isinstance(artwork, RasterArtwork):
        _, offset_x_mm, offset_y_mm, content_width_mm, content_height_mm = _content_size_mm(
            artwork, spec
        )
        px_per_mm = spec.px_per_mm
        target_w = max(1, round(content_width_mm * px_per_mm))
        target_h = max(1, round(content_height_mm * px_per_mm))
        resized = _resample_raster(artwork.image, target_w, target_h)
        x0 = round(offset_x_mm * px_per_mm)
        y0 = round(offset_y_mm * px_per_mm)
        x1 = min(page_width_px, x0 + target_w)
        y1 = min(page_height_px, y0 + target_h)
        canvas[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    else:
        transform = fit_transform(artwork, spec)
        px_per_mm = spec.px_per_mm
        thickness = max(1, round(spec.stroke_width_mm * px_per_mm))
        for path in artwork.paths:
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
