"""Abstract interface that every conversion engine must implement.

New engines (classical or ML-based) plug into the CLI and pipeline purely
through this interface, so the rest of the codebase never needs to know
which concrete engine produced a given line-art image.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


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
    return a single-channel (grayscale) image of the same height and
    width, where dark pixels are lines and light pixels are the paper
    background. Keeping this contract narrow is what lets the CLI treat
    every engine identically regardless of the technique behind it.
    """

    #: Short, CLI-facing identifier for this engine (e.g. "canny").
    name: str

    @abstractmethod
    def convert(
        self, image: np.ndarray, *, line_thickness: int = 1, debug: DebugSink | None = None
    ) -> np.ndarray:
        """Convert a BGR photo into a grayscale coloring-page line drawing.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR order with shape ``(H, W, 3)`` and dtype
            ``uint8``, as returned by ``cv2.imread``.
        line_thickness : int, optional
            Approximate line thickness in pixels, by default ``1``. Engines
            should use this to dilate/scale their detected edges so results
            stay comparable across styles.
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
        np.ndarray
            Single-channel image with shape ``(H, W)`` and dtype ``uint8``,
            where ``255`` is background (white paper) and lower values are
            line art, suitable for printing.
        """
        raise NotImplementedError
