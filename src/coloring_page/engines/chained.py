"""Flatten-chain-redraw conversion engine using cv2.ximgproc."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from coloring_page.drawing import Drawing, paths_from_point_chains, rasterize
from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.pipeline import derive_kernel_size


def detect_edge_chains(
    image: np.ndarray,
    *,
    sigma_color: float = 35.0,
    sigma_space: float = 6.0,
    num_iterations: int = 4,
    gradient_threshold: float = 36.0,
    anchor_threshold: float = 8.0,
    min_path_length: int = 30,
    edge_sigma: float = 1.5,
    nfa_validation: bool = True,
    debug: DebugSink | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Flatten texture and detect connected edge chains via EdgeDrawing.

    The first two stages of :class:`ChainedEngine`, split out so other
    engines (e.g. :class:`~coloring_page.engines.gated.GatedEngine`) can
    reuse the same flatten+chain detection without duplicating it or
    running a full :class:`ChainedEngine` conversion just to get its
    intermediate edge chains.

    Parameters
    ----------
    image : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
    sigma_color, sigma_space, num_iterations, gradient_threshold,
    anchor_threshold, min_path_length, edge_sigma, nfa_validation :
        See :class:`ChainedEngine`'s constructor for what each parameter
        controls.
    debug : DebugSink | None, optional
        If given, the flattened image and edge-chain map are saved as
        debug stages.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        The flattened image's grayscale conversion, and the list of edge
        chain segments (each an ``Nx1x2`` int32 point array) from
        ``cv2.ximgproc.EdgeDrawing.getSegments()``.
    """
    flattened = cv2.ximgproc.rollingGuidanceFilter(
        image, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space, numOfIter=num_iterations
    )
    if debug is not None:
        debug.save("flattened", flattened)
    gray = cv2.cvtColor(flattened, cv2.COLOR_BGR2GRAY)

    edge_drawing = cv2.ximgproc.createEdgeDrawing()
    params = cv2.ximgproc.EdgeDrawing.Params()
    params.GradientThresholdValue = int(gradient_threshold)
    params.AnchorThresholdValue = int(anchor_threshold)
    params.MinPathLength = min_path_length
    params.Sigma = edge_sigma
    params.NFAValidation = nfa_validation
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

    return gray, segments


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

    Was the CLI's default until ``scripts/ablation.py`` measured it
    against ``lineart-raster`` on this project's reference photos: this
    engine's ink coverage came in 2-3x over the target band, with
    hundreds of noise-sized "enclosed regions" per image (texture and
    paper grain getting chained into spurious closed loops, not real
    colorable areas) -- see
    :class:`~coloring_page.engines.lineart_raster.LineArtRasterEngine`'s
    docstring and the README for the numbers. Kept `experimental` and
    registered for comparison, not as a coloring-page candidate.
    """

    name = "chained"
    experimental = True

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

    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> Drawing:
        """Flatten texture, chain edges, then trace them as vector paths.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        gray, segments = detect_edge_chains(
            image,
            sigma_color=self.sigma_color,
            sigma_space=self.sigma_space,
            num_iterations=self.num_iterations,
            gradient_threshold=self.gradient_threshold,
            anchor_threshold=self.anchor_threshold,
            min_path_length=self.min_path_length,
            edge_sigma=self.edge_sigma,
            nfa_validation=self.nfa_validation,
            debug=debug,
        )

        # EdgeDrawing's chains are already exact traced geometry --
        # wrapping them directly (rather than rasterizing to a mask and
        # re-tracing with paths_from_mask) avoids quantizing precision
        # the source already has.
        paths = paths_from_point_chains(segments, gray.shape)
        drawing = Drawing(paths=paths, aspect_ratio=image.shape[1] / image.shape[0])
        drawing = drawing.simplify(self.polyline_epsilon / max(gray.shape[:2]))

        working_dimension = max(gray.shape[:2])
        min_length = derive_kernel_size(working_dimension, fraction=0.012, min_value=3, odd=False)
        min_length_normalized = min_length / working_dimension
        drawing = drawing.filter(lambda p: p.length >= min_length_normalized)

        if debug is not None:
            debug.save("redraw", rasterize(drawing, long_side_px=working_dimension))
        return drawing
