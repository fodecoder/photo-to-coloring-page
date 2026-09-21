"""Abstract interface that every conversion engine must implement.

New engines (classical or ML-based) plug into the CLI and pipeline purely
through this interface, so the rest of the codebase never needs to know
which concrete engine produced a given line-art image.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from coloring_page.drawing import Drawing


class DebugSink(Protocol):
    """Receives an engine's intermediate stages for inspection.

    Defined as a structural (``Protocol``) type here, rather than a
    concrete class, so this module -- the shared base every engine
    already depends on -- doesn't need to import the concrete
    file-writing implementation from ``pipeline.py``, which in turn
    depends on this module. Any object with a matching ``save`` method
    (see :class:`coloring_page.pipeline.DebugSink`) satisfies this type.
    """

    def save(self, stage_name: str, image: np.ndarray) -> None:
        """Record one named intermediate image."""
        ...


class ConversionEngine(ABC):
    """Base class for algorithms that turn a photo into line art.

    Implementations receive a BGR image (as loaded by OpenCV) and must
    return a :class:`~coloring_page.drawing.Drawing`: a normalized vector
    representation of the traced line art, decoupled from any particular
    raster resolution -- see :mod:`coloring_page.drawing` for the full
    contract. Keeping this contract narrow is what lets the CLI treat
    every engine identically regardless of the technique behind it.
    """

    #: Short, CLI-facing identifier for this engine (e.g. "canny").
    name: str

    #: Whether this engine is a superseded baseline kept only for
    #: comparison, not a candidate default. Experimental engines are
    #: hidden from ``--help`` unless ``--show-experimental`` is passed,
    #: but remain fully usable via ``--style <name>``.
    experimental: bool = False

    #: True for engines that hold a model resident in (V)RAM across
    #: calls. Batch conversion (:mod:`coloring_page.batch`) would
    #: otherwise instantiate one such engine per worker process under
    #: ``ProcessPoolExecutor``, multiplying VRAM/RAM use and risking OOM
    #: on a single GPU; batch execution forces ``jobs=1`` for these
    #: regardless of the requested job count.
    requires_serial_execution: bool = False

    @abstractmethod
    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> Drawing:
        """Convert a BGR photo into vector coloring-page line art.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR order with shape ``(H, W, 3)`` and dtype
            ``uint8``, as returned by ``cv2.imread``.
        debug : DebugSink | None, optional
            When given, engines with multiple internal stages (a
            flatten/filter step, an edge map, a redraw pass, ...) may call
            ``debug.save(stage_name, image)`` after each one, so a
            developer can inspect which stage produced an unexpected
            result. By default None, in which case engines must skip
            this reporting entirely (not merely no-op) since there is no
            image size/dtype contract on ``debug.save``'s calls -- it is
            purely diagnostic. Engines with no meaningful internal stages
            may ignore this parameter.

        Returns
        -------
        Drawing
            Normalized vector line art, decoupled from any raster
            resolution -- see :mod:`coloring_page.drawing`.
        """
        raise NotImplementedError
