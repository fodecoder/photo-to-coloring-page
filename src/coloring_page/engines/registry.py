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
    from coloring_page.engines.lineart import LineArtEngine
except ImportError:
    # The `lineart` extra (torch, sam2, controlnet_aux) isn't installed --
    # this engine simply doesn't appear as a `--style` choice unless
    # `pip install -e ".[lineart]"` was run.
    pass
else:
    ENGINES["lineart"] = LineArtEngine


def get_engine(style: str, *, device: str = "auto") -> ConversionEngine:
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
    if "device" in inspect.signature(engine_cls.__init__).parameters:
        return engine_cls(device=device)  # type: ignore[call-arg]
    return engine_cls()
