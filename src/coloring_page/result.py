"""``Result``: what :func:`coloring_page.convert_image` returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coloring_page.artwork import Artwork
from coloring_page.page import PageSpec
from coloring_page.pipeline import save_image
from coloring_page.render import to_pdf, to_png, to_svg
from coloring_page.validate import QualityReport


@dataclass(frozen=True)
class Result:
    """The output of one :func:`coloring_page.convert_image` call.

    Attributes
    ----------
    drawing : Artwork
        The final line art (after detail filtering, for a ``Drawing``) --
        either a :class:`~coloring_page.drawing.Drawing` or a
        :class:`~coloring_page.artwork.RasterArtwork`, see
        :mod:`coloring_page.artwork`. Kept named ``drawing`` rather than
        renamed to ``artwork`` to avoid touching every existing caller for
        a cosmetic rename.
    report : QualityReport
        Colorability measurements for ``drawing`` rendered onto ``page``.
        Always present -- see :func:`coloring_page.convert_image`'s
        docstring for why a failing report doesn't itself raise.
    timings : dict[str, float]
        Per-stage duration in seconds, plus a ``"total"`` key.
    page : PageSpec
        The page geometry ``drawing`` was validated (and would be
        rendered) against.

    This is the only place in the package that performs output file I/O
    -- no :class:`~coloring_page.engines.base.ConversionEngine` writes to
    disk.
    """

    drawing: Artwork
    report: QualityReport
    timings: dict[str, float]
    page: PageSpec

    def save_svg(self, path: Path) -> None:
        """Write ``drawing`` as an SVG document to ``path``, creating parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_svg(self.drawing, self.page), encoding="utf-8")

    def save_pdf(self, path: Path) -> None:
        """Write ``drawing`` as a single-page PDF to ``path``, creating parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(to_pdf(self.drawing, self.page))

    def save_png(self, path: Path) -> None:
        """Write ``drawing`` as a full-page raster PNG to ``path``, creating parent dirs."""
        save_image(to_png(self.drawing, self.page), path)
