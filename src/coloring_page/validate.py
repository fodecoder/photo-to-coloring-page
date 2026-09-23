"""Quantitative colorability checks for a rendered coloring page.

Nothing upstream of this module ever checked whether a traced contour is
actually *closed* -- the main reason generated output has been unusable as
a real coloring page: an "enclosed region" a marker can be dropped into is
exactly what an open contour with a gap fails to provide. This replaces
the project's old acceptance criterion, ``boundary_f_measure`` (which
compared output to an artistic reference image and, per
``docs/DIAGNOSIS.md`` §4, rewarded drawing more ink rather than drawing
the *right*, colorable ink -- see ``scripts/metrics.py``, where that old
metric now lives as a diagnostic-only dev tool). :func:`validate` measures
colorability directly instead of measuring similarity to a reference.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np

from coloring_page.artwork import Artwork, RasterArtwork
from coloring_page.drawing import Drawing
from coloring_page.page import PageSpec
from coloring_page.render import to_png

#: Minimum ink-component size, in pixels, below which a flood-fill
#: "leak" candidate is treated as antialiasing noise rather than a real
#: gap. Small relative to any real leaked region, comfortably above the
#: 1-2px slivers LINE_AA stroke rendering can leave at path joins.
_LEAK_NOISE_FLOOR_PX = 4


def ink_coverage(binary_image: np.ndarray) -> float:
    """Fraction of pixels that are ink rather than background.

    This is the project's canonical ink-fraction metric -- every other
    module (including ``scripts/metrics.py``'s diagnostic tooling) that
    needs an ink-coverage number imports it from here rather than
    recomputing it.

    Parameters
    ----------
    binary_image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background (paper) and darker pixels are ink.

    Returns
    -------
    float
        Ink pixel fraction in ``[0, 1]``. :class:`QualityReport`'s default
        admissible band is 3-8%, the range a printable coloring page
        needs to land in to be neither a near-blank page nor illegibly
        dense.
    """
    return float(np.mean(binary_image < 128))


def antialiased_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that are neither near-black ink nor near-white paper.

    A :class:`~coloring_page.artwork.RasterArtwork` is only worth having
    instead of a binarized mask if it actually preserves antialiasing and
    stroke-width modulation -- this is the check that catches an engine
    that silently thresholded its own output before returning it (in which
    case this fraction collapses to near 0).

    Parameters
    ----------
    image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``.

    Returns
    -------
    float
        Fraction of pixels strictly between 5 and 250, in ``[0, 1]``. The
        margins on both ends exclude ordinary print-black/paper-white
        noise from counting as "antialiasing."
    """
    return float(np.mean((image > 5) & (image < 250)))


def _flood_fill_from_border(ink_mask: np.ndarray) -> np.ndarray:
    """Flood-fill background from the canvas border, padded so every edge is seeded.

    Parameters
    ----------
    ink_mask : np.ndarray
        Boolean array, ``True`` where a pixel is ink.

    Returns
    -------
    np.ndarray
        ``uint8`` array, same shape as ``ink_mask``: ``128`` where
        background was reached by the border flood-fill, ``255`` where
        background was never reached (enclosed), ``0`` where ``ink_mask``
        is ``True``.
    """
    background = np.where(ink_mask, np.uint8(0), np.uint8(255))
    padded = cv2.copyMakeBorder(background, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)

    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(
        padded,
        flood_mask,
        seedPoint=(0, 0),
        newVal=128,
        loDiff=0,
        upDiff=0,
        flags=4,
    )

    return padded[1:-1, 1:-1]


@dataclass(frozen=True)
class QualityReport:
    """Colorability measurements for one rendered page, plus pass/fail thresholds.

    A single report type serves both a :class:`~coloring_page.drawing.Drawing`
    and a :class:`~coloring_page.artwork.RasterArtwork`, rather than two
    distinct report types, because the two only disagree on two fields:
    ``leaking_regions`` and ``dangling_endpoints`` are both concepts that
    only make sense for traced paths with an exact ``closed``/``open``
    ground truth (see :func:`validate`) -- a raster has no such ground
    truth, so both are reported as ``None`` (not a failed ``0``) for a
    ``RasterArtwork``, and :attr:`passed` treats ``None`` as "not
    applicable, doesn't gate" for either field, the same way it already
    does for ``min_region_area_mm2``.

    Attributes
    ----------
    ink_coverage : float
        See :func:`ink_coverage`.
    enclosed_regions : int
        Count of background regions the border flood-fill never reaches
        -- the areas actually colorable with a marker or crayon.
    leaking_regions : int | None
        Count of background regions that appear meant to be enclosed
        (they would be, if every path were fully closed) but the border
        flood-fill reaches anyway, through a gap in the contour. See
        :func:`validate` for the detection algorithm. ``None`` for a
        ``RasterArtwork`` -- not applicable, not a failure.
    min_region_area_mm2 : float | None
        Smallest enclosed region's area in mm^2, or ``None`` if there are
        no enclosed regions.
    region_area_mm2_p5 : float | None
        5th percentile of enclosed region areas in mm^2, or ``None`` if
        there are no enclosed regions.
    dangling_endpoints : int | None
        Count of open-path endpoints (``2`` per open ``Path``, ``0`` per
        closed one) -- every dangling endpoint is a place a contour fails
        to close. ``None`` for a ``RasterArtwork`` -- not applicable, not
        a failure.
    antialiased_fraction : float | None
        See :func:`antialiased_fraction`. ``None`` for a
        :class:`~coloring_page.drawing.Drawing`, which has no raster
        antialiasing of its own to measure until rendered with a fixed
        stroke width.
    ink_coverage_range : tuple[float, float]
        Admissible ``(min, max)`` ink coverage band, by default
        ``(0.03, 0.08)``.
    max_leaking_regions : int
        Largest ``leaking_regions`` value still considered passing, by
        default ``0``. Not checked when ``leaking_regions`` is ``None``.
    max_dangling_endpoints : int
        Largest ``dangling_endpoints`` value still considered passing, by
        default ``0``. Not checked when ``dangling_endpoints`` is ``None``.
    min_region_area_mm2_floor : float
        Smallest ``min_region_area_mm2`` still considered passing (a
        region below this is too small to color with a marker), by
        default ``1.0``.
    min_antialiased_fraction : float
        Smallest ``antialiased_fraction`` still considered passing, by
        default ``0.05``. Not checked when ``antialiased_fraction`` is
        ``None``.
    """

    ink_coverage: float
    enclosed_regions: int
    leaking_regions: int | None
    min_region_area_mm2: float | None
    region_area_mm2_p5: float | None
    dangling_endpoints: int | None
    antialiased_fraction: float | None = None
    ink_coverage_range: tuple[float, float] = (0.03, 0.08)
    max_leaking_regions: int = 0
    max_dangling_endpoints: int = 0
    min_region_area_mm2_floor: float = 1.0
    min_antialiased_fraction: float = 0.05

    @property
    def passed(self) -> bool:
        """Whether every measurement falls within its configured threshold.

        Returns
        -------
        bool
        """
        min_ink, max_ink = self.ink_coverage_range
        ink_ok = min_ink <= self.ink_coverage <= max_ink
        leaking_ok = (
            self.leaking_regions is None or self.leaking_regions <= self.max_leaking_regions
        )
        endpoints_ok = (
            self.dangling_endpoints is None
            or self.dangling_endpoints <= self.max_dangling_endpoints
        )
        area_ok = (
            self.min_region_area_mm2 is None
            or self.min_region_area_mm2 >= self.min_region_area_mm2_floor
        )
        antialiased_ok = (
            self.antialiased_fraction is None
            or self.antialiased_fraction >= self.min_antialiased_fraction
        )
        return ink_ok and leaking_ok and endpoints_ok and area_ok and antialiased_ok


#: Threshold defaults for a :class:`~coloring_page.drawing.Drawing` --
#: today's original thresholds. ``leaking_regions=0``/``dangling_endpoints=0``
#: are only meaningful because a vector path has exact closed/open ground
#: truth; see :data:`RASTER_THRESHOLDS` for why a raster can't use them.
VECTOR_THRESHOLDS: dict[str, object] = {
    "ink_coverage_range": (0.03, 0.08),
    "max_leaking_regions": 0,
    "max_dangling_endpoints": 0,
    "min_region_area_mm2_floor": 1.0,
}

#: Threshold defaults for a :class:`~coloring_page.artwork.RasterArtwork`.
#: Omits ``max_leaking_regions``/``max_dangling_endpoints`` -- those fields
#: are reported as ``None`` for a raster (see :class:`QualityReport`), so
#: their thresholds are moot -- and adds ``min_antialiased_fraction``, the
#: raster-only check that an engine hasn't silently binarized its output.
#: ``ink_coverage_range`` reuses the vector band, calibrated against a real
#: reference line-art image's own measured statistics (~4.7% full-black
#: ink -- see tests/test_validate_calibration.py), deliberately NOT against
#: whatever a given raster engine currently happens to produce: a
#: threshold that rejects its own ideal is broken (this repo's lesson,
#: twice already), but loosening it to match an engine's current output
#: instead of a real target is the same mistake in the other direction.
#: ``lineart-raster`` at its current default settings measures well below
#: this band on real photos (~0.02-0.8%, see ``scripts/ablation.py`` and
#: ``python scripts/report_quality.py docs --style lineart-raster``) --
#: that's a disclosed, real limitation of this engine's current ink
#: density (see its README section), not a reason to relax this band.
RASTER_THRESHOLDS: dict[str, object] = {
    "ink_coverage_range": (0.03, 0.08),
    "min_region_area_mm2_floor": 1.0,
    "min_antialiased_fraction": 0.05,
}


def _validate_drawing(drawing: Drawing, spec: PageSpec) -> dict[str, object]:
    """Compute the vector-only measurements: leaking regions and dangling endpoints.

    Split out of :func:`validate` so that function's dispatch stays a
    thin ``isinstance`` branch rather than one long function mixing both
    representations' measurement logic.
    """
    raster = to_png(drawing, spec)
    ink_mask = raster < 128

    flood_result = _flood_fill_from_border(ink_mask)
    enclosed_mask = flood_result == 255
    reached_mask = flood_result == 128

    num_enclosed, _, enclosed_stats, _ = cv2.connectedComponentsWithStats(
        enclosed_mask.astype(np.uint8), connectivity=8
    )
    enclosed_regions = num_enclosed - 1

    mm2_per_px2 = (25.4 / spec.dpi) ** 2
    if enclosed_regions > 0:
        areas_mm2 = enclosed_stats[1:, cv2.CC_STAT_AREA] * mm2_per_px2
        min_region_area_mm2 = float(areas_mm2.min())
        region_area_mm2_p5 = float(np.percentile(areas_mm2, 5))
    else:
        min_region_area_mm2 = None
        region_area_mm2_p5 = None

    # A "leaking" region is background that looks like it was meant to be
    # enclosed -- it would be, if every path were fully closed -- but a
    # gap lets the border flood-fill reach it anyway. Force-closing every
    # path turns a small real gap into a short closing chord that seals
    # the contour (cv2.polylines(isClosed=True) already draws that
    # segment), so re-running the same flood-fill on the force-closed
    # raster reveals which currently-"reached" background would have been
    # enclosed. This is deliberately a vector-space heuristic rather than
    # a fixed-radius morphological close: the gap-bridging distance is
    # driven by the path's own geometry, not an arbitrary pixel kernel
    # size. It's weaker for gaps wide enough that the closing chord
    # meaningfully changes the enclosed shape (e.g. a half-open circle),
    # but that's not the failure mode this check targets -- a small gap
    # in otherwise-closed line art is.
    closed_paths = tuple(dataclasses.replace(path, closed=True) for path in drawing.paths)
    closed_drawing = dataclasses.replace(drawing, paths=closed_paths)
    closed_raster = to_png(closed_drawing, spec)
    closed_flood_result = _flood_fill_from_border(closed_raster < 128)
    enclosed_mask_if_closed = closed_flood_result == 255

    leak_pixels = reached_mask & enclosed_mask_if_closed
    _, _, leak_stats, _ = cv2.connectedComponentsWithStats(
        leak_pixels.astype(np.uint8), connectivity=8
    )
    leaking_regions = int(np.sum(leak_stats[1:, cv2.CC_STAT_AREA] >= _LEAK_NOISE_FLOOR_PX))

    # Path.closed is exact ground truth for whether a contour has a gap --
    # an open Path has exactly 2 endpoints (its first and last point) by
    # construction, a closed one has none. Re-skeletonizing the raster to
    # rediscover this would be strictly less precise (resolution-dependent,
    # can spuriously merge/split near-touching strokes) and would duplicate
    # information the vector model already has exactly. Known limitation:
    # two distinct open Paths whose endpoints happen to coincide spatially
    # (e.g. a chain split at a junction during tracing) still count as 4
    # dangling endpoints here rather than being recognized as meeting.
    dangling_endpoints = sum(2 for path in drawing.paths if not path.closed)

    return {
        "ink_coverage": ink_coverage(raster),
        "enclosed_regions": enclosed_regions,
        "leaking_regions": leaking_regions,
        "min_region_area_mm2": min_region_area_mm2,
        "region_area_mm2_p5": region_area_mm2_p5,
        "dangling_endpoints": dangling_endpoints,
        "antialiased_fraction": None,
    }


def _validate_raster(artwork: RasterArtwork, spec: PageSpec) -> dict[str, object]:
    """Compute the raster-only measurements: enclosed regions after a 1px morphological closing.

    A raster has no vector path geometry to force-close, so
    ``leaking_regions``/``dangling_endpoints`` aren't computed at all here
    (:func:`validate` sets them to ``None``). A small morphological closing
    is used instead, purely to bridge the sub-pixel antialiasing gaps a
    raster line naturally has at its own edges -- unlike
    :func:`_validate_drawing`'s vector-space gap heuristic, there is no
    path geometry here to reason about, so a fixed small kernel is the
    only tool available.
    """
    raster = to_png(artwork, spec)
    ink_mask = (raster < 128).astype(np.uint8)
    closed_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    flood_result = _flood_fill_from_border(closed_mask.astype(bool))
    enclosed_mask = flood_result == 255

    num_enclosed, _, enclosed_stats, _ = cv2.connectedComponentsWithStats(
        enclosed_mask.astype(np.uint8), connectivity=8
    )
    enclosed_regions = num_enclosed - 1

    mm2_per_px2 = (25.4 / spec.dpi) ** 2
    if enclosed_regions > 0:
        areas_mm2 = enclosed_stats[1:, cv2.CC_STAT_AREA] * mm2_per_px2
        min_region_area_mm2 = float(areas_mm2.min())
        region_area_mm2_p5 = float(np.percentile(areas_mm2, 5))
    else:
        min_region_area_mm2 = None
        region_area_mm2_p5 = None

    return {
        "ink_coverage": ink_coverage(raster),
        "enclosed_regions": enclosed_regions,
        "leaking_regions": None,
        "min_region_area_mm2": min_region_area_mm2,
        "region_area_mm2_p5": region_area_mm2_p5,
        "dangling_endpoints": None,
        "antialiased_fraction": antialiased_fraction(raster),
    }


def validate(artwork: Artwork, spec: PageSpec, **threshold_overrides: object) -> QualityReport:
    """Measure ``artwork``'s colorability once rendered onto ``spec``.

    Dispatches on whether ``artwork`` is a
    :class:`~coloring_page.drawing.Drawing` or a
    :class:`~coloring_page.artwork.RasterArtwork` -- see
    :class:`QualityReport` for how the two differ, and
    :data:`VECTOR_THRESHOLDS`/:data:`RASTER_THRESHOLDS` for the threshold
    defaults each uses when ``threshold_overrides`` doesn't say otherwise.

    Parameters
    ----------
    artwork : Artwork
        The line art to validate.
    spec : PageSpec
        The page geometry ``artwork`` would be printed onto -- rendering
        resolution (``spec.dpi``) and, for a ``Drawing``, physical stroke
        width both affect the measurements, since a thicker stroke seals a
        gap a thinner one wouldn't.
    **threshold_overrides : object
        Passed through to override any of :class:`QualityReport`'s
        threshold fields, on top of :data:`VECTOR_THRESHOLDS`/
        :data:`RASTER_THRESHOLDS`'s defaults for ``artwork``'s type.

    Returns
    -------
    QualityReport
    """
    if isinstance(artwork, RasterArtwork):
        measurements = _validate_raster(artwork, spec)
        thresholds: dict[str, object] = {**RASTER_THRESHOLDS, **threshold_overrides}
    else:
        measurements = _validate_drawing(artwork, spec)
        thresholds = {**VECTOR_THRESHOLDS, **threshold_overrides}

    return QualityReport(**measurements, **thresholds)  # type: ignore[arg-type]
