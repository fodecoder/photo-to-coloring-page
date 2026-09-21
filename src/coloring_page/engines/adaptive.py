"""Adaptive-thresholding conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.drawing import Drawing, paths_from_mask
from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.pipeline import smooth_preserving_edges


class AdaptiveEngine(ConversionEngine):
    """Produce a textured pencil-sketch effect with adaptive thresholding.

    This mimics a classic "pencil sketch" technique: the image is
    heavily blurred, then thresholded per-neighborhood so that local
    contrast (rather than a single global cutoff) decides where a line
    falls. It preserves shading detail as texture rather than clean
    outlines, which on real photos produces far too much ink coverage
    and too many small, disconnected fragments to work as a coloring
    page -- prefer ``canny``, ``xdog``, ``chained``, or ``skeleton`` for
    that use case. This style is kept for users who want the textured
    look for its own sake.
    """

    name = "adaptive"
    experimental = True

    def __init__(self, block_size: int = 9, c: int = 2) -> None:
        """Store the adaptive-threshold neighborhood size and constant.

        Parameters
        ----------
        block_size : int, optional
            Size (in pixels) of the pixel neighborhood used to compute each
            threshold, by default 9. Must be odd; even values are bumped up
            by one.
        c : int, optional
            Constant subtracted from the computed mean/weighted mean, by
            default 2. Higher values produce thinner, sparser lines.
        """
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c = c

    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> Drawing:
        """Blur then adaptively threshold the image into vector line art.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = smooth_preserving_edges(gray)
        lines = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c,
        )
        if debug is not None:
            debug.save("thresholded", lines)

        # adaptiveThreshold's own convention is ink=0/background=255 --
        # the opposite of paths_from_mask's nonzero=ink contract.
        ink_mask = cv2.bitwise_not(lines)
        paths = paths_from_mask(ink_mask)
        working_dimension = max(gray.shape[:2])
        min_length_normalized = 4 / working_dimension
        drawing = Drawing(paths=paths, aspect_ratio=image.shape[1] / image.shape[0])
        return drawing.filter(lambda p: p.length >= min_length_normalized)
