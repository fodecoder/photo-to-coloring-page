"""Semantic-gated conversion engine: chained's geometry, a network's selection.

Requires the ``ml`` extra (``pip install -e ".[ml]"``) for ``torch``, and
``informative_drawings``' pretrained weights -- see that engine's module
docstring and the README's "Extending with an ML engine" section.
"""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.engines.chained import detect_edge_chains
from coloring_page.engines.informative_drawings import InformativeDrawingsEngine
from coloring_page.pipeline import derive_kernel_size, remove_short_strokes
from coloring_page.postprocess import normalize_percentile, redraw_segments


class GatedEngine(ConversionEngine):
    """Keep only ``chained``'s edge chains a pretrained network is confident about.

    Neither ingredient alone solves the over-inking problem this project
    is built around: ``chained``'s EdgeDrawing stage already recalls most
    of a reference drawing's real strokes (see the project's own
    measurements), but a gradient-magnitude threshold can't tell a real
    object outline from water-ripple or foliage texture with the same
    local contrast, so precision stays low no matter how the threshold is
    tuned. ``informative_drawings`` was trained to make exactly that
    semantic distinction, but its own soft output has the same texture
    sensitivity once binarized on its own -- see
    ``coloring_page.postprocess``'s measured results.

    This engine combines them instead of choosing one: run EdgeDrawing
    permissively (favoring recall) to get connected edge *chains*, run
    the network to get a per-pixel confidence map on the same image, then
    keep only the chains the network is confident lie on real ink,
    discarding the rest before redrawing. The classical stage supplies
    already-connected, already-smooth geometry; the network supplies the
    semantic selection neither one can do alone.

    **Measured result**: on this project's three reference pairs, this
    engine does not currently beat plain ``chained``. Gating does raise
    precision meaningfully as the threshold rises, but recall falls
    faster than precision rises at every threshold tested, so average
    boundary F1 is highest at ``gate_threshold=0`` (a no-op -- every
    chain survives, since confidence can't be negative) and declines
    from there. The per-chain *mean* confidence used here is a plausible
    first attempt, not a validated design; a max- or percentile-based
    aggregate, or a per-pixel rather than per-chain gate, might do
    better and hasn't been tried. Kept registered for experimentation
    (see ``scripts/compare.py``), excluded from
    ``scripts/compare.py``'s ``RECOMMENDED_STYLES``.
    """

    name = "gated"

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
        gate_threshold: float = 40.0,
        tolerance_radius: int = 2,
        informative_drawings: InformativeDrawingsEngine | None = None,
    ) -> None:
        """Store the gating parameters and the flatten/chain stage's own parameters.

        Parameters
        ----------
        sigma_color, sigma_space, num_iterations, gradient_threshold,
        anchor_threshold, min_path_length, edge_sigma, nfa_validation,
        polyline_epsilon :
            Passed through to
            :func:`~coloring_page.engines.chained.detect_edge_chains` and
            the redraw stage; see
            :class:`~coloring_page.engines.chained.ChainedEngine`'s
            constructor for what each one controls. Defaults match
            ``ChainedEngine``'s own defaults (a permissive starting
            point), since gating is meant to recover the precision a
            tighter threshold would otherwise have to sacrifice recall
            for.
        gate_threshold : float, optional
            Minimum network confidence (on a normalized 0-255 scale,
            255 = maximally ink-like) an edge chain's average sampled
            response must reach to survive, by default 40.0. Note that
            confidence can never be negative, so ``0.0`` is a no-op:
            every chain always survives, making the engine identical to
            plain ``chained``. Measured against this project's three
            reference pairs, raising the threshold *does* raise
            precision substantially (0.55-0.85 at 40.0, up to 0.6-0.9 by
            100.0) but recall falls faster than precision rises at every
            value tested -- average boundary F1 is highest at the no-op
            point and declines monotonically from there, meaning
            mean-confidence gating does not currently beat plain
            ``chained`` on this data at any real (non-zero) threshold.
            40.0 is kept as the default anyway, as the least-bad
            non-degenerate setting, rather than defaulting to the no-op
            that would silently make this engine pointless. See the
            README's "Conversion styles" section and
            ``scripts/compare.py`` before relying on this engine.
        tolerance_radius : int, optional
            Pixel radius used to dilate each chain before sampling the
            confidence map under it, by default 2. Accounts for small
            misalignment between EdgeDrawing's traced geometry and the
            network's own (resized/upsampled) output.
        informative_drawings : InformativeDrawingsEngine | None, optional
            The network instance to gate against. If None (the default),
            a new one is constructed with its own default weights
            resolution. Passing an existing instance lets its (lazily
            loaded) weights be shared across multiple conversions.
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
        self.gate_threshold = gate_threshold
        self.tolerance_radius = tolerance_radius
        self._informative_drawings = informative_drawings or InformativeDrawingsEngine()

    def _segment_confidence(self, segment: np.ndarray, normalized_soft: np.ndarray) -> float:
        """Average network confidence sampled under a dilated edge chain.

        Parameters
        ----------
        segment : np.ndarray
            One EdgeDrawing chain, an ``Nx1x2`` or ``Nx2`` int32 point array.
        normalized_soft : np.ndarray
            Percentile-normalized network soft map, same ``(H, W)`` as
            the source image.

        Returns
        -------
        float
            Mean confidence (0-255, higher = more ink-like) of
            ``normalized_soft`` under the chain, widened by
            ``self.tolerance_radius``. ``0.0`` if the chain covers no
            pixels (shouldn't happen for a real chain, but guards
            against division by zero on a degenerate one).
        """
        mask = np.zeros(normalized_soft.shape, dtype=np.uint8)
        cv2.polylines(mask, [segment], isClosed=False, color=255, thickness=1)
        if self.tolerance_radius > 0:
            kernel_size = 2 * self.tolerance_radius + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel).astype(np.uint8)

        sampled = normalized_soft[mask > 0]
        if sampled.size == 0:
            return 0.0
        return float(255 - sampled.mean())

    def convert(
        self, image: np.ndarray, *, line_thickness: int = 1, debug: DebugSink | None = None
    ) -> np.ndarray:
        """Detect edge chains, gate them by network confidence, then redraw survivors.

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

        soft = self._informative_drawings.soft_map(image)
        normalized_soft = normalize_percentile(soft)
        if debug is not None:
            debug.save("soft_map", normalized_soft)

        kept_segments = [
            segment
            for segment in segments
            if self._segment_confidence(segment, normalized_soft) >= self.gate_threshold
        ]
        if debug is not None:
            gated_map = redraw_segments(
                kept_segments, gray.shape, polyline_epsilon=self.polyline_epsilon
            )
            debug.save("gated_chains", gated_map)

        canvas = redraw_segments(
            kept_segments,
            gray.shape,
            polyline_epsilon=self.polyline_epsilon,
            line_thickness=line_thickness,
        )
        if debug is not None:
            debug.save("redraw", canvas)

        working_dimension = max(gray.shape[:2])
        min_extent = derive_kernel_size(working_dimension, fraction=0.012, min_value=3, odd=False)
        return remove_short_strokes(canvas, min_extent=min_extent)
