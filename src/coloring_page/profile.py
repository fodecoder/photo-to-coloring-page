"""``Profile``: the product-facing configuration for one conversion.

Before this module, a conversion's "quality knob" was ``--thickness``, a
pixel line width with no relationship to the printed page. ``detail`` is
the real product parameter: how much of a source photo's fine structure
ends up as colorable line art depends on who's going to color it, not on
what resolution the source photo happened to be. See :func:`apply_detail`
for how a ``detail`` level is turned into an actual filter over an
engine's output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from coloring_page.page import PageSpec

if TYPE_CHECKING:
    from coloring_page.drawing import Drawing
    from coloring_page.engines.base import DebugSink

#: Public detail levels, in increasing order of fine detail retained.
DetailLevel = Literal["toddler", "child", "adult"]


@dataclass(frozen=True)
class Profile:
    """Everything :func:`coloring_page.convert_image` needs beyond the input image.

    Attributes
    ----------
    page : PageSpec
        Target page geometry (size, margins, print resolution, stroke
        width) -- see :mod:`coloring_page.page`.
    style : str
        A key of ``coloring_page.engines.registry.ENGINES`` (e.g.
        ``"chained"``).
    detail : {"toddler", "child", "adult"}
        How much fine structure survives into the output -- see
        :data:`DETAIL_PRESETS` and :func:`apply_detail`. By default
        ``"child"``.
    device : {"cpu", "cuda", "auto"}
        Inference device for ML-backed styles; ignored by classical
        (non-ML) styles. By default ``"auto"`` (CUDA if available, else
        CPU).
    seed : int | None
        Passed to :func:`coloring_page.seeding.seed_everything`. By
        default ``0``, for reproducible-by-default conversions; pass
        ``None`` for nondeterministic behavior.
    max_pixels : int
        Resource limit enforced by :func:`coloring_page.api.convert_image`
        before any conversion work happens, by default 40,000,000 (~40
        megapixels).
    debug_dir : Path | None
        If set, intermediate conversion stages (both the engine's own and
        the detail-filter pre/post rasters) are written here. By default
        None.
    """

    page: PageSpec = field(default_factory=PageSpec)
    style: str = "chained"
    detail: DetailLevel = "child"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    seed: int | None = 0
    max_pixels: int = 40_000_000
    debug_dir: Path | None = None

    def to_json(self) -> str:
        """Serialize to a JSON string.

        Returns
        -------
        str
        """
        data = {
            "page": self.page.to_dict(),
            "style": self.style,
            "detail": self.detail,
            "device": self.device,
            "seed": self.seed,
            "max_pixels": self.max_pixels,
            "debug_dir": str(self.debug_dir) if self.debug_dir is not None else None,
        }
        return json.dumps(data)

    @staticmethod
    def from_json(data: str) -> Profile:
        """Reconstruct a ``Profile`` from :meth:`to_json`'s output.

        Parameters
        ----------
        data : str

        Returns
        -------
        Profile

        Raises
        ------
        ValueError
            If ``detail`` is present but not one of ``"toddler"``,
            ``"child"``, ``"adult"``.
        """
        parsed = json.loads(data)
        detail = parsed.get("detail", "child")
        if detail not in ("toddler", "child", "adult"):
            raise ValueError(f"Unknown detail level {detail!r}; expected toddler/child/adult.")
        debug_dir = parsed.get("debug_dir")
        return Profile(
            page=PageSpec.from_dict(parsed["page"]) if "page" in parsed else PageSpec(),
            style=parsed.get("style", "chained"),
            detail=detail,
            device=parsed.get("device", "auto"),
            seed=parsed.get("seed", 0),
            max_pixels=parsed.get("max_pixels", 40_000_000),
            debug_dir=Path(debug_dir) if debug_dir is not None else None,
        )

    def with_overrides(self, **changes: object) -> Profile:
        """Return a copy with the given fields replaced.

        A thin wrapper over ``dataclasses.replace`` so callers (chiefly
        the CLI, layering per-invocation flags onto a ``--profile`` file)
        don't need to import ``dataclasses`` themselves.

        Parameters
        ----------
        **changes : object
            Field name/value pairs to override.

        Returns
        -------
        Profile
        """
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class DetailParams:
    """Engine-agnostic thresholds one ``detail`` level maps to.

    Attributes
    ----------
    min_region_area_mm2 : float
        Minimum area (at print size) of the smaller region a boundary
        path separates, below which the path is dropped -- a no-op for
        paths with no ``source_area`` (see :attr:`coloring_page.drawing.Path.source_area`).
    min_stroke_length_mm : float
        Minimum path length (at print size), below which the path is
        dropped.
    simplify_epsilon_mm : float
        ``cv2.approxPolyDP`` tolerance (at print size) applied to every
        surviving path.
    """

    min_region_area_mm2: float
    min_stroke_length_mm: float
    simplify_epsilon_mm: float


#: Preset thresholds per detail level. Values chosen so ``toddler`` keeps
#: only large, simple shapes (few big regions, thick minimum stroke
#: length, aggressive simplification) and ``adult`` keeps most of an
#: engine's native output (small area/length floors, light
#: simplification). Not empirically tuned against reference images yet --
#: see the README's profile table for the intended visual effect of each
#: level and its documented limitation for engines with no region concept.
DETAIL_PRESETS: dict[DetailLevel, DetailParams] = {
    "toddler": DetailParams(
        min_region_area_mm2=300.0, min_stroke_length_mm=6.0, simplify_epsilon_mm=1.0
    ),
    "child": DetailParams(
        min_region_area_mm2=120.0, min_stroke_length_mm=3.0, simplify_epsilon_mm=0.6
    ),
    "adult": DetailParams(
        min_region_area_mm2=25.0, min_stroke_length_mm=1.5, simplify_epsilon_mm=0.3
    ),
}

#: Inference resolution (short side, px) a raster-style engine constructed
#: with a ``resolution`` parameter should run at for a given detail level --
#: separate from :data:`DETAIL_PRESETS`'s mm-based post-hoc filter
#: thresholds, since a raster engine has no post-hoc filter step to apply
#: those to (see :func:`apply_detail`). A CNN run at half the resolution
#: has a proportionally larger receptive field relative to the scene, so a
#: lower resolution draws less detail and proportionally thicker strokes --
#: "simplify" for a raster engine is a resolution change, not a filter.
RASTER_RESOLUTION_PRESETS: dict[DetailLevel, int] = {
    "toddler": 512,
    "child": 768,
    "adult": 1280,
}


def _content_area_mm2(scale_mm_per_unit: float, aspect_ratio: float) -> float:
    """Physical area, in mm^2, that a ``Drawing``'s normalized unit square maps to.

    Parameters
    ----------
    scale_mm_per_unit : float
        As returned by :func:`coloring_page.page.fit_transform`.
    aspect_ratio : float
        ``Drawing.aspect_ratio``.

    Returns
    -------
    float
    """
    if aspect_ratio >= 1:
        width_units, height_units = 1.0, 1.0 / aspect_ratio
    else:
        width_units, height_units = aspect_ratio, 1.0
    return (width_units * scale_mm_per_unit) * (height_units * scale_mm_per_unit)


def apply_detail(drawing: Drawing, profile: Profile, *, debug: DebugSink | None = None) -> Drawing:
    """Filter/simplify an engine's output to match ``profile.detail``.

    No :class:`~coloring_page.engines.base.ConversionEngine` needs to know
    about ``detail`` at all: this is a post-hoc pass over whatever
    ``Drawing`` an engine already produced, using
    :meth:`~coloring_page.drawing.Drawing.filter` and
    :meth:`~coloring_page.drawing.Drawing.simplify` plus
    :func:`~coloring_page.page.fit_transform`'s scale to convert this
    preset's millimeter thresholds into ``Drawing``'s normalized units.

    Engines that never populate :attr:`~coloring_page.drawing.Path.source_area`
    (everything except ``region``/``gated``) degrade gracefully: their
    area filter is a no-op (every path passes), so ``toddler``'s "few
    large regions" effect on those engines comes entirely from the
    stroke-length and simplification thresholds -- an approximation, not
    a true area filter, documented in the README's profile table.

    Parameters
    ----------
    drawing : Drawing
        The engine's raw output.
    profile : Profile
        Supplies ``detail`` and ``page`` (needed to convert mm thresholds
        to normalized units).
    debug : DebugSink | None, optional
        Unused directly here; accepted so callers can pass the same sink
        they use elsewhere without a conditional, by default None.

    Returns
    -------
    Drawing
    """
    del debug
    from coloring_page.page import fit_transform

    params = DETAIL_PRESETS[profile.detail]
    scale_mm_per_unit, _, _ = fit_transform(drawing, profile.page)
    content_area_mm2 = _content_area_mm2(scale_mm_per_unit, drawing.aspect_ratio)

    min_length_norm = params.min_stroke_length_mm / scale_mm_per_unit
    drawing = drawing.filter(lambda p: p.length >= min_length_norm)

    if content_area_mm2 > 0:
        min_area_norm = params.min_region_area_mm2 / content_area_mm2
        drawing = drawing.filter(lambda p: p.source_area is None or p.source_area >= min_area_norm)

    epsilon_norm = params.simplify_epsilon_mm / scale_mm_per_unit
    return drawing.simplify(epsilon_norm)
