"""Abstract interface that every conversion engine must implement.

New engines (classical or ML-based) plug into the CLI and pipeline purely
through this interface, so the rest of the codebase never needs to know
which concrete engine produced a given line-art image.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


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
    def convert(self, image: np.ndarray, *, line_thickness: int = 1) -> np.ndarray:
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

        Returns
        -------
        np.ndarray
            Single-channel image with shape ``(H, W)`` and dtype ``uint8``,
            where ``255`` is background (white paper) and lower values are
            line art, suitable for printing.
        """
        raise NotImplementedError
