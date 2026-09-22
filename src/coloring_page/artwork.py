"""``Artwork``: the two representations a conversion engine may return.

:mod:`coloring_page.drawing`'s vector model is not always achievable. It
buys closed contours (:attr:`~coloring_page.drawing.Path.closed` is exact
ground truth, not inferred) and a stroke width that stays a fixed physical
size (mm) at any output resolution -- but only by tracing raster ink into
geometry, and tracing requires deciding where ink starts and stops. A soft
detector response with antialiased edges and stroke-width modulation baked
into its grayscale gradients has no such crisp boundary: binarizing it to
get one throws away exactly the modulation that made it look drawn by hand,
and re-tracing the binarized result can shatter a continuous stroke into
many short, disconnected fragments (each becoming a spurious pair of
:attr:`~coloring_page.drawing.Path` endpoints). For that kind of output, an
antialiased raster *is* the best achievable coloring page -- not a
degraded stand-in for a vector one.

:class:`RasterArtwork` is that second representation, kept deliberately
close to :class:`~coloring_page.drawing.Drawing` in shape (an image plus an
aspect ratio plus free-form metadata) so :data:`Artwork` can stand in for
either wherever an engine's output flows downstream (rendering,
validation). Neither representation is strictly better than the other:
:class:`~coloring_page.drawing.Drawing` is the right choice whenever an
engine can trace real closed contours; :class:`RasterArtwork` is the right
choice whenever forcing that trace would destroy the very ink pattern that
made the output good.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from coloring_page.drawing import Drawing


@dataclass(frozen=True)
class RasterArtwork:
    """A finished coloring page as an antialiased grayscale raster, not traced geometry.

    Attributes
    ----------
    image : np.ndarray
        Single-channel ``uint8`` image, shape ``(H, W)``. ``0`` is full
        ink, ``255`` is white paper, and values in between are meaningful
        -- unlike :func:`coloring_page.drawing.rasterize`'s output, this is
        deliberately *not* binarized. A producing engine must not threshold
        its own antialiasing away before returning it.
    aspect_ratio : float
        ``width / height`` of :attr:`image`, same convention as
        :attr:`coloring_page.drawing.Drawing.aspect_ratio` -- this is what
        lets :func:`coloring_page.page.fit_transform` place either
        representation on a page identically.
    meta : Mapping[str, object]
        Free-form provenance/debugging metadata (e.g. which engine
        produced this, what inference resolution it ran at). Not
        interpreted by anything in this module.

    ``RasterArtwork`` performs no validation of its own -- it is a thin,
    trusting container, matching :class:`coloring_page.drawing.Path` and
    :class:`coloring_page.drawing.Drawing`'s own convention. A producing
    engine is responsible for only ever constructing valid instances.
    """

    image: np.ndarray
    aspect_ratio: float
    meta: Mapping[str, object] = field(default_factory=dict)


#: What a :class:`~coloring_page.engines.base.ConversionEngine` may return:
#: traced vector geometry, or an antialiased raster -- see the module
#: docstring for when each is the right choice.
Artwork: TypeAlias = Drawing | RasterArtwork
