"""Print-page geometry: physical page size, margins, and content placement.

Everything upstream of this module (engines, ``Drawing``) works in
normalized ``[0, 1]``-ish coordinates decoupled from any output resolution
-- see :mod:`coloring_page.drawing`. :class:`PageSpec` is where that
abstract geometry finally meets a physical page: a size in millimeters, a
margin, a print resolution, and a physical stroke width. Every renderer in
:mod:`coloring_page.render` shares the same placement math
(:func:`fit_transform`) so an SVG, a PDF, and a PNG of the same
``(drawing, spec)`` pair always agree pixel-for-pixel (well, mm-for-mm) on
where content sits on the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from coloring_page.artwork import Artwork

#: Named page sizes in millimeters, as ``(width, height)`` in portrait
#: orientation. ``LETTER`` uses the US customary 8.5x11in size converted to
#: mm (215.9 x 279.4).
_PAGE_SIZES_MM: dict[str, tuple[float, float]] = {
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "LETTER": (215.9, 279.4),
}


@dataclass(frozen=True)
class PageSpec:
    """Physical page geometry a ``Drawing`` is rendered/printed onto.

    Attributes
    ----------
    size : {"A4", "A5", "LETTER"} | tuple[float, float]
        A named page size, or an explicit ``(width_mm, height_mm)`` pair.
        A named size is resolved via :data:`_PAGE_SIZES_MM`; an explicit
        tuple is used exactly as given, regardless of :attr:`orientation`
        -- orientation only rotates a *named* preset, since a caller who
        already specified explicit dimensions has already made that
        choice.
    orientation : {"portrait", "landscape"}
        For a named :attr:`size`, whether to use it as-is (portrait) or
        with width/height swapped (landscape). By default "portrait".
    margin_mm : float
        Blank border kept on every edge of the page, by default 10.0.
    dpi : int
        Print/raster resolution in dots per inch, by default 300 --
        standard print quality, well above what a marker or crayon needs
        to resolve but common enough that most home/office printers
        default to it.
    stroke_width_mm : float
        Physical width of a drawn line, by default 0.7 -- comparable to a
        fine-tip marker, thick enough to stay visible after printing
        shrinks anything thinner than a printer's own dot gain.
    """

    size: Literal["A4", "A5", "LETTER"] | tuple[float, float] = "A4"
    orientation: Literal["portrait", "landscape"] = "portrait"
    margin_mm: float = 10.0
    dpi: int = 300
    stroke_width_mm: float = 0.7

    @property
    def page_size_mm(self) -> tuple[float, float]:
        """Resolved ``(width_mm, height_mm)`` of the page.

        Returns
        -------
        tuple[float, float]

        Raises
        ------
        ValueError
            If :attr:`size` is an explicit tuple that isn't a pair of
            positive numbers, or a string that isn't a known preset name.
        """
        if isinstance(self.size, str):
            try:
                width, height = _PAGE_SIZES_MM[self.size]
            except KeyError:
                known = ", ".join(sorted(_PAGE_SIZES_MM))
                raise ValueError(f"Unknown page size {self.size!r}. Known sizes: {known}") from None
            if self.orientation == "landscape":
                width, height = height, width
            return width, height

        width, height = self.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Page size must be a pair of positive mm values, got {self.size!r}")
        return float(width), float(height)

    @property
    def usable_size_mm(self) -> tuple[float, float]:
        """``page_size_mm`` minus :attr:`margin_mm` on every edge.

        Returns
        -------
        tuple[float, float]
            ``(usable_width_mm, usable_height_mm)``.

        Raises
        ------
        ValueError
            If the margins leave no usable area (or a negative one).
        """
        page_width, page_height = self.page_size_mm
        usable_width = page_width - 2 * self.margin_mm
        usable_height = page_height - 2 * self.margin_mm
        if usable_width <= 0 or usable_height <= 0:
            raise ValueError(
                f"margin_mm={self.margin_mm} leaves no usable area on a "
                f"{page_width}x{page_height}mm page."
            )
        return usable_width, usable_height

    @property
    def px_per_mm(self) -> float:
        """Pixels per millimeter at :attr:`dpi` (``dpi / 25.4``)."""
        return self.dpi / 25.4

    @property
    def page_size_px(self) -> tuple[int, int]:
        """Resolved page size in pixels at :attr:`dpi`, as ``(width, height)``."""
        page_width_mm, page_height_mm = self.page_size_mm
        px_per_mm = self.px_per_mm
        return max(1, round(page_width_mm * px_per_mm)), max(1, round(page_height_mm * px_per_mm))

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible ``dict``.

        Returns
        -------
        dict[str, object]
            ``size`` is emitted as a list (``[width_mm, height_mm]``) when
            it's an explicit tuple, or unchanged when it's a named preset
            string -- :meth:`from_dict` distinguishes the two cases the
            same way on the way back in.
        """
        return {
            "size": list(self.size) if not isinstance(self.size, str) else self.size,
            "orientation": self.orientation,
            "margin_mm": self.margin_mm,
            "dpi": self.dpi,
            "stroke_width_mm": self.stroke_width_mm,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> PageSpec:
        """Reconstruct a ``PageSpec`` from :meth:`to_dict`'s output.

        Parameters
        ----------
        data : dict[str, object]

        Returns
        -------
        PageSpec
        """
        raw_size = data["size"]
        size: Literal["A4", "A5", "LETTER"] | tuple[float, float]
        if isinstance(raw_size, str):
            size = cast(Literal["A4", "A5", "LETTER"], raw_size)
        else:
            raw_pair = cast("list[object]", raw_size)
            size = (
                float(cast("str | int | float", raw_pair[0])),
                float(cast("str | int | float", raw_pair[1])),
            )
        return PageSpec(
            size=size,
            orientation=cast(Literal["portrait", "landscape"], data["orientation"]),
            margin_mm=float(cast("str | int | float", data["margin_mm"])),
            dpi=int(cast("str | int | float", data["dpi"])),
            stroke_width_mm=float(cast("str | int | float", data["stroke_width_mm"])),
        )


def fit_transform(artwork: Artwork, spec: PageSpec) -> tuple[float, float, float]:
    """Compute the mapping from an ``Artwork``'s normalized coordinates to page mm.

    Every renderer in :mod:`coloring_page.render` calls this instead of
    deriving its own placement, so an SVG/PDF/PNG export of the same
    ``(artwork, spec)`` pair always places content identically.

    The content is inscribed into :attr:`PageSpec.usable_size_mm`
    preserving its aspect ratio (letterboxed, not stretched or cropped)
    and centered within it. Fitting is done against ``artwork.aspect_ratio``
    -- the aspect ratio of the *source image* the content was normalized
    against, not any bounding box of the ink itself -- so placement doesn't
    shift depending on how much of the frame happens to contain ink. This
    only reads ``.aspect_ratio``, so it works identically for a
    :class:`~coloring_page.drawing.Drawing` or a
    :class:`~coloring_page.artwork.RasterArtwork`.

    Parameters
    ----------
    artwork : Artwork
        The line art to place.
    spec : PageSpec
        The target page geometry.

    Returns
    -------
    tuple[float, float, float]
        ``(scale_mm_per_unit, offset_x_mm, offset_y_mm)``. A normalized
        point ``(x, y)`` maps to page mm via
        ``(offset_x_mm + x * scale, offset_y_mm + y * scale)`` -- both
        axes share the same ``scale``, which is what preserves aspect
        ratio.
    """
    if artwork.aspect_ratio >= 1:
        content_width_units, content_height_units = 1.0, 1.0 / artwork.aspect_ratio
    else:
        content_width_units, content_height_units = artwork.aspect_ratio, 1.0

    usable_width_mm, usable_height_mm = spec.usable_size_mm
    scale = min(usable_width_mm / content_width_units, usable_height_mm / content_height_units)

    content_width_mm = content_width_units * scale
    content_height_mm = content_height_units * scale
    offset_x_mm = spec.margin_mm + (usable_width_mm - content_width_mm) / 2
    offset_y_mm = spec.margin_mm + (usable_height_mm - content_height_mm) / 2

    return scale, offset_x_mm, offset_y_mm
