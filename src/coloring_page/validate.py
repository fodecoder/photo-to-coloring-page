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

    Attributes
    ----------
    ink_coverage : float
        See :func:`ink_coverage`.
    enclosed_regions : int
        Count of background regions the border flood-fill never reaches
        -- the areas actually colorable with a marker or crayon.
    leaking_regions : int
        Count of background regions that appear meant to be enclosed
        (they would be, if every path were fully closed) but the border
        flood-fill reaches anyway, through a gap in the contour. See
        :func:`validate` for the detection algorithm.
    min_region_area_mm2 : float | None
        Smallest enclosed region's area in mm^2, or ``None`` if there are
        no enclosed regions.
    region_area_mm2_p5 : float | None
        5th percentile of enclosed region areas in mm^2, or ``None`` if
        there are no enclosed regions.
    dangling_endpoints : int
        Count of open-path endpoints (``2`` per open ``Path``, ``0`` per
        closed one) -- every dangling endpoint is a place a contour fails
        to close.
    ink_coverage_range : tuple[float, float]
        Admissible ``(min, max)`` ink coverage band, by default
        ``(0.03, 0.08)``.
    max_leaking_regions : int
        Largest ``leaking_regions`` value still considered passing, by
        default ``0``.
    max_dangling_endpoints : int
        Largest ``dangling_endpoints`` value still considered passing, by
        default ``0``.
    min_region_area_mm2_floor : float
        Smallest ``min_region_area_mm2`` still considered passing (a
        region below this is too small to color with a marker), by
        default ``1.0``.
    """

    ink_coverage: float
    enclosed_regions: int
    leaking_regions: int
    min_region_area_mm2: float | None
    region_area_mm2_p5: float | None
    dangling_endpoints: int
    ink_coverage_range: tuple[float, float] = (0.03, 0.08)
    max_leaking_regions: int = 0
    max_dangling_endpoints: int = 0
    min_region_area_mm2_floor: float = 1.0

    @property
    def passed(self) -> bool:
        """Whether every measurement falls within its configured threshold.

        Returns
        -------
        bool
        """
        min_ink, max_ink = self.ink_coverage_range
        ink_ok = min_ink <= self.ink_coverage <= max_ink
        leaking_ok = self.leaking_regions <= self.max_leaking_regions
        endpoints_ok = self.dangling_endpoints <= self.max_dangling_endpoints
        area_ok = (
            self.min_region_area_mm2 is None
            or self.min_region_area_mm2 >= self.min_region_area_mm2_floor
        )
        return ink_ok and leaking_ok and endpoints_ok and area_ok


def validate(drawing: Drawing, spec: PageSpec, **threshold_overrides: object) -> QualityReport:
    """Measure ``drawing``'s colorability once rendered onto ``spec``.

    Parameters
    ----------
    drawing : Drawing
        The vector line art to validate.
    spec : PageSpec
        The page geometry ``drawing`` would be printed onto -- rendering
        resolution (``spec.dpi``) and physical stroke width both affect
        the measurements, since a thicker stroke seals a gap a thinner one
        wouldn't.
    **threshold_overrides : object
        Passed through to override any of :class:`QualityReport`'s
        threshold fields (``ink_coverage_range``, ``max_leaking_regions``,
        ``max_dangling_endpoints``, ``min_region_area_mm2_floor``).

    Returns
    -------
    QualityReport
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

    return QualityReport(
        ink_coverage=ink_coverage(raster),
        enclosed_regions=enclosed_regions,
        leaking_regions=leaking_regions,
        min_region_area_mm2=min_region_area_mm2,
        region_area_mm2_p5=region_area_mm2_p5,
        dangling_endpoints=dangling_endpoints,
        **threshold_overrides,  # type: ignore[arg-type]
    )
