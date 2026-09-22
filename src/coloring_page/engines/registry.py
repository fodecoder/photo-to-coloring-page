"""Maps CLI-facing style names to their conversion engine classes.

Adding a new engine? Register it in ``ENGINES`` below and it will
automatically become available as a ``--style`` choice in the CLI. See
``.claude/skills/add-conversion-style/SKILL.md`` for the full checklist.
"""

from __future__ import annotations

import inspect

from coloring_page.engines.adaptive import AdaptiveEngine
from coloring_page.engines.base import ConversionEngine
from coloring_page.engines.canny import CannyEngine
from coloring_page.engines.cartoon import CartoonEngine
from coloring_page.engines.chained import ChainedEngine
from coloring_page.engines.region import RegionEngine
from coloring_page.engines.skeleton_redraw import SkeletonRedrawEngine
from coloring_page.engines.xdog import XDoGEngine
from coloring_page.exceptions import EngineUnavailableError

ENGINES: dict[str, type[ConversionEngine]] = {
    "canny": CannyEngine,
    "adaptive": AdaptiveEngine,
    "xdog": XDoGEngine,
    "cartoon": CartoonEngine,
    "chained": ChainedEngine,
    "skeleton": SkeletonRedrawEngine,
    "region": RegionEngine,
}

try:
    from coloring_page.engines.anime2sketch import Anime2SketchEngine
except ImportError:
    # The `ml` extra (torch) isn't installed -- the default install has
    # zero ML dependencies, so this engine simply doesn't appear as a
    # `--style` choice unless `pip install -e ".[ml]"` was run.
    pass
else:
    ENGINES["anime2sketch"] = Anime2SketchEngine

try:
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine
except ImportError:
    pass
else:
    ENGINES["informative_drawings"] = InformativeDrawingsEngine

    from coloring_page.engines.gated import GatedEngine

    ENGINES["gated"] = GatedEngine

try:
    from coloring_page.engines.lineart_raster import LineArtRasterEngine
except ImportError:
    # The `lineart_raster` extra (torch, controlnet_aux) isn't installed --
    # this engine simply doesn't appear as a `--style` choice unless
    # `pip install -e ".[lineart_raster]"` was run. A separate try/except
    # from `lineart` below (rather than sharing one) since this engine
    # doesn't need `sam2` -- it must stay available even when `sam2` isn't
    # installed.
    pass
else:
    ENGINES["lineart-raster"] = LineArtRasterEngine

try:
    from coloring_page.engines.lineart import LineArtEngine
except ImportError:
    # The `lineart` extra (torch, sam2, controlnet_aux) isn't installed --
    # this engine simply doesn't appear as a `--style` choice unless
    # `pip install -e ".[lineart]"` was run.
    pass
else:
    ENGINES["lineart"] = LineArtEngine


def get_engine(
    style: str, *, device: str = "auto", resolution: int | None = None
) -> ConversionEngine:
    """Instantiate the conversion engine registered under ``style``.

    Parameters
    ----------
    style : str
        One of the keys in ``ENGINES`` (e.g. ``"canny"``).
    device : str, optional
        Inference device, by default ``"auto"``. Passed to the engine's
        constructor only when it accepts a ``device`` parameter (checked
        via ``inspect.signature``) -- classical (non-ML) engines don't,
        and are constructed with no arguments regardless of this value.
    resolution : int | None, optional
        Inference resolution, by default None. Passed to the engine's
        constructor only when it accepts a ``resolution`` parameter --
        the reserved name a raster-style engine adopts to opt into
        ``Profile.detail``-driven resolution wiring (see
        ``coloring_page.profile.RASTER_RESOLUTION_PRESETS`` and
        ``coloring_page.engines.lineart_raster.LineArtRasterEngine``).
        ``lineart.py``'s ``segmentation_resolution``/``detail_resolution``
        deliberately use different names and so are not affected by this.

    Returns
    -------
    ConversionEngine
        A freshly constructed engine instance using its default parameters.

    Raises
    ------
    EngineUnavailableError
        If ``style`` is not a registered engine name (either unknown
        outright, or registered only behind an optional extra that isn't
        installed).
    """
    try:
        engine_cls = ENGINES[style]
    except KeyError as exc:
        available = ", ".join(sorted(ENGINES))
        raise EngineUnavailableError(
            f"Unknown style {style!r}. Available styles: {available}"
        ) from exc
    params = inspect.signature(engine_cls.__init__).parameters
    kwargs: dict[str, object] = {}
    if "device" in params:
        kwargs["device"] = device
    if resolution is not None and "resolution" in params:
        kwargs["resolution"] = resolution
    return engine_cls(**kwargs)
