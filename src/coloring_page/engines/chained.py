"""Flatten-chain-redraw conversion engine using cv2.ximgproc."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.pipeline import derive_kernel_size, remove_short_strokes


class ChainedEngine(ConversionEngine):
    """Produce line art via flatten -> chain -> redraw, instead of mask edits.

    ``canny``, ``adaptive``, and ``xdog`` all react to local intensity
    *gradients*, and ``cartoon``'s region boundaries still come from
    thresholding a gradient image -- so all four end up editing a raw
    pixel *mask*: dilating it, closing small gaps, filtering specks. A
    mask edit can bridge a one-pixel gap, but it can't turn a hundred
    short, jittery mask fragments back into the handful of long, smooth
    strokes a hand-drawn coloring page actually has.

    This engine instead works in three explicit stages:

    1. **Flatten** -- ``cv2.ximgproc.rollingGuidanceFilter`` (Zhang et
       al. 2014) removes small-scale texture (brush strokes, paper
       grain, soft shading) while preserving large-scale object
       boundaries, directly attacking the source of the noise rather
       than cleaning it up after the fact.
    2. **Chain** -- ``cv2.ximgproc.createEdgeDrawing`` returns connected
       edge *chains* (``getSegments()``), not a pixel mask. This solves
       broken contours at the source: there's no gap to bridge, and
       chains below ``min_path_length`` can be filtered out directly
       instead of via post-hoc component-size filtering.
    3. **Redraw** -- each chain is smoothed with ``cv2.approxPolyDP``
       and redrawn with ``cv2.polylines`` on a blank canvas. This is
       what produces a uniform stroke width: the *geometry* is redrawn,
       not the mask.
    """

    name = "chained"

    def __init__(
        self,
        sigma_color: float = 35.0,
        sigma_space: float = 6.0,
        num_iterations: int = 4,
        gradient_threshold: float = 36.0,
        anchor_threshold: float = 8.0,
        min_path_length: int = 30,
        edge_sigma: float = 1.5,
        nfa_validation: bool = True,
        polyline_epsilon: float = 1.2,
    ) -> None:
        """Store the flatten/chain/redraw stage parameters.

        Parameters
        ----------
        sigma_color : float, optional
            Rolling guidance filter's color-space sigma, by default
            35.0. Larger values merge more distant colors together,
            flattening more texture.
        sigma_space : float, optional
            Rolling guidance filter's coordinate-space sigma, by default
            6.0.
        num_iterations : int, optional
            Number of rolling guidance filter iterations, by default 4.
            More iterations flatten more aggressively.
        gradient_threshold : float, optional
            EdgeDrawing's gradient magnitude threshold for anchor/edge
            pixels, by default 36.0.
        anchor_threshold : float, optional
            EdgeDrawing's threshold for placing anchor points, by
            default 8.0.
        min_path_length : int, optional
            Minimum edge chain length, in pixels, for EdgeDrawing to
            keep it, by default 30. This is where most residual noise
            gets filtered, directly on the chain rather than on pixels.
        edge_sigma : float, optional
            Gaussian smoothing sigma EdgeDrawing applies before edge
            detection, by default 1.5.
        nfa_validation : bool, optional
            Whether EdgeDrawing validates candidate edges with a
            number-of-false-alarms test to reject spurious ones, by
            default True.
        polyline_epsilon : float, optional
            ``cv2.approxPolyDP`` tolerance used to smooth each chain
            before redrawing, in pixels, by default 1.2. Larger values
            give straighter, less jittery strokes at the cost of fine
            detail.
        """
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.num_iterations = num_iterations
        self.gradient_threshold = gradient_threshold
        self.anchor_threshold = anchor_threshold
        self.min_path_length = min_path_length
        self.edge_sigma = edge_sigma
        self.nfa_validation = nfa_validation
        self.polyline_epsilon = polyline_epsilon

    def convert(
        self, image: np.ndarray, *, line_thickness: int = 1, debug: DebugSink | None = None
    ) -> np.ndarray:
        """Flatten texture, chain edges, then redraw them as smooth strokes.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        flattened = cv2.ximgproc.rollingGuidanceFilter(
            image,
            d=-1,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space,
            numOfIter=self.num_iterations,
        )
        if debug is not None:
            debug.save("flattened", flattened)
        gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)

        edge_drawing = cv2.ximgproc.createEdgeDrawing()
        params = cv2.ximgproc.EdgeDrawing.Params()
        params.GradientThresholdValue = int(self.gradient_threshold)
        params.AnchorThresholdValue = int(self.anchor_threshold)
        params.MinPathLength = self.min_path_length
        params.Sigma = self.edge_sigma
        params.NFAValidation = self.nfa_validation
        edge_drawing.setParams(params)
        edge_drawing.detectEdges(gray)
        # cv2's stub types getSegments() as Sequence[Sequence[Point]], but it
        # actually returns a tuple of Nx2 int32 ndarrays; cast to what
        # approxPolyDP/polylines actually need.
        segments = cast("list[np.ndarray]", edge_drawing.getSegments())

        if debug is not None:
            edge_map = np.full(gray.shape, 255, dtype=np.uint8)
            cv2.polylines(edge_map, list(segments), isClosed=False, color=0, thickness=1)
            debug.save("edge_chains", edge_map)

        canvas = np.full(gray.shape, 255, dtype=np.uint8)
        for segment in segments:
            approx = cv2.approxPolyDP(segment, self.polyline_epsilon, closed=False)
            cv2.polylines(
                canvas, [approx], isClosed=False, color=0, thickness=line_thickness,
                lineType=cv2.LINE_AA,
            )
        if debug is not None:
            debug.save("redraw", canvas)

        working_dimension = max(gray.shape[:2])
        min_extent = derive_kernel_size(working_dimension, fraction=0.012, min_value=3, odd=False)
        return remove_short_strokes(canvas, min_extent=min_extent)
