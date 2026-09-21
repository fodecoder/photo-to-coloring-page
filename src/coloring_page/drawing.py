"""Vector line-art domain model: paths instead of pixels.

Every conversion engine used to pass raster arrays between stages and
return a raster bitmap. That conflates two resolutions that are actually
different things: the *working* resolution an engine happens to detect
edges at (see :data:`coloring_page.pipeline.DEFAULT_WORKING_DIMENSION`),
and the *output* resolution a finished coloring page is eventually printed
or exported at. Baking the first into the returned pixels is what made
stroke thickness a function of working resolution, made a closed contour
an accident of how a particular raster operation happened to trace it
rather than a guarantee, and left no way to express "this region is too
small to be colored with a marker" once everything was already flattened
to pixels.

:class:`Drawing` and :class:`Path` fix this by representing line art as
geometry, not pixels. A :class:`Path`'s points are normalized to ``[0, 1]``
relative to the *long side* of whatever image the engine actually ran on,
so a :class:`Drawing` can be rendered at any output resolution later
without re-running detection, and comparing/filtering paths never depends
on what working resolution happened to produce them.

Coordinate convention
----------------------
- Points are ``(x, y)`` -- matches OpenCV's own convention
  (``cv2.findContours``, ``cv2.polylines``, ``EdgeDrawing.getSegments()``
  all use ``(x, y)``), so the adapters below need no axis swap except
  where a computation is naturally ``(row, col)`` (the skeleton walk in
  :func:`paths_from_mask`).
- Both axes are normalized by the same scalar, ``long_side = max(H, W)``,
  not each by its own dimension -- this is what makes the stored points
  preserve the source image's aspect ratio.
- :attr:`Drawing.aspect_ratio` is ``width / height`` of the image the
  points were normalized against. To reconstruct pixel coordinates for a
  target ``long_side_px``: derive the output canvas as
  ``(long_side_px, round(long_side_px / aspect_ratio))`` when
  ``aspect_ratio >= 1`` (width is the long side), or the transposed pair
  otherwise; then ``pixel_xy = normalized_xy * long_side_px``. See
  :func:`rasterize` for a working implementation of this formula.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

#: 8-neighborhood sum kernel (excludes the center pixel itself). Used with
#: ``cv2.filter2D`` to classify a skeleton pixel by how many of its 8
#: neighbors are also skeleton: 1 = endpoint, 2 = interior, >=3 = junction.
#: This module is the shared home for these skeleton-graph primitives;
#: ``postprocess.prune_short_branches`` imports them from here rather than
#: defining its own copy.
NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

#: Offsets, in ``(dy, dx)`` order, to each of a pixel's 8 neighbors.
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]  # fmt: skip


@dataclass(frozen=True, eq=False)
class Path:
    """One traced line, as an ordered sequence of normalized points.

    ``eq=False``: a dataclass's generated ``__eq__`` compares fields as a
    tuple, and comparing two multi-element ``np.ndarray``s with ``==``
    raises ``ValueError`` when Python tries to coerce the result to a
    ``bool`` -- a real runtime trap for a field of this type, not
    something ``mypy``/``ruff`` would catch. ``Path`` instances don't
    support ``==``; compare ``.points`` directly with
    ``np.testing.assert_array_equal`` instead.

    Attributes
    ----------
    points : np.ndarray
        Shape ``(N, 2)``, dtype ``float32``, normalized ``(x, y)``
        coordinates in ``[0, 1]`` -- see the module docstring for the
        normalization convention. Always ``N >= 2``; producing fewer
        points than that is a bug in whatever adapter created this
        ``Path``, not a valid degenerate case.
    closed : bool
        Whether the path loops back from its last point to its first
        (e.g. a region boundary that forms a closed ring) rather than
        being an open stroke.
    kind : {"boundary", "detail"}
        Whether this path traces the edge between two regions
        (``"boundary"``) or an internal line that doesn't correspond to a
        region boundary (``"detail"``, e.g. a facial feature line from an
        ML engine's soft map).
    source_area : float | None
        Normalized area (as a fraction of the total image area, in
        ``[0, 1]``) of the smaller of the two regions adjacent to this
        path, when the producing engine has that information (currently
        only :class:`~coloring_page.engines.region.RegionEngine`). ``None``
        when the producing engine has no region concept. Feeds a
        downstream saliency filter: a boundary between two tiny regions is
        a candidate to drop even if the path itself is long.

    ``Path`` performs no validation of its own -- it is a thin, trusting
    container. :func:`paths_from_mask` and :func:`paths_from_point_chains`
    are responsible for only ever constructing valid instances.
    """

    points: np.ndarray
    closed: bool
    kind: Literal["boundary", "detail"]
    source_area: float | None = None

    @property
    def length(self) -> float:
        """Total path length in normalized units.

        Sum of consecutive-point Euclidean distances, plus the closing
        segment (last point back to the first) when :attr:`closed` is
        True. This is the primitive every engine uses in place of the old
        raster-only ``pipeline.remove_short_strokes``: a
        :class:`Drawing` is pruned of noise via
        ``drawing.filter(lambda p: p.length >= min_length)`` instead.
        """
        segments = np.diff(self.points, axis=0)
        total = float(np.linalg.norm(segments, axis=1).sum())
        if self.closed and len(self.points) > 1:
            total += float(np.linalg.norm(self.points[0] - self.points[-1]))
        return total


@dataclass(frozen=True)
class Drawing:
    """A complete piece of vector line art: every traced path, plus enough context to render it.

    Attributes
    ----------
    paths : tuple[Path, ...]
        Every traced line making up this drawing.
    aspect_ratio : float
        ``width / height`` of the image :attr:`paths`'s points were
        normalized against -- see the module docstring's coordinate
        convention for how a renderer uses this to reconstruct pixel
        coordinates at any output resolution.
    meta : Mapping[str, object]
        Free-form provenance/debugging metadata (e.g. which engine
        produced this, what working resolution it ran at). Not
        interpreted by anything in this module.
    """

    paths: tuple[Path, ...]
    aspect_ratio: float
    meta: Mapping[str, object] = field(default_factory=dict)

    def filter(self, predicate: Callable[[Path], bool]) -> Drawing:
        """Return a new ``Drawing`` keeping only paths matching ``predicate``.

        Parameters
        ----------
        predicate : Callable[[Path], bool]
            Called once per path; a path is kept when this returns True.

        Returns
        -------
        Drawing
            A new instance with the same ``aspect_ratio``/``meta``.
        """
        return Drawing(
            paths=tuple(path for path in self.paths if predicate(path)),
            aspect_ratio=self.aspect_ratio,
            meta=self.meta,
        )

    def simplify(self, epsilon: float) -> Drawing:
        """Return a new ``Drawing`` with every path polyline-simplified.

        Parameters
        ----------
        epsilon : float
            ``cv2.approxPolyDP`` tolerance, in the same normalized
            ``[0, 1]`` units as :attr:`Path.points` -- this keeps
            ``Drawing`` fully resolution-agnostic. A caller wanting "1.2px
            at 1400px working resolution" (a typical classical-engine
            ``polyline_epsilon``) passes ``epsilon=1.2 / 1400``.

        Returns
        -------
        Drawing
            A new instance with the same ``aspect_ratio``/``meta``.
        """
        simplified = []
        for path in self.paths:
            approx = cv2.approxPolyDP(path.points, epsilon, closed=path.closed)
            simplified.append(
                Path(
                    points=approx.reshape(-1, 2).astype(np.float32),
                    closed=path.closed,
                    kind=path.kind,
                    source_area=path.source_area,
                )
            )
        return Drawing(paths=tuple(simplified), aspect_ratio=self.aspect_ratio, meta=self.meta)

    def bounds(self) -> tuple[float, float, float, float]:
        """Bounding box across every path's points, in normalized units.

        Returns
        -------
        tuple[float, float, float, float]
            ``(min_x, min_y, max_x, max_y)``.

        Raises
        ------
        ValueError
            If this ``Drawing`` has no paths -- bounds are undefined, and
            a real ``Drawing`` always has at least one path, so an empty
            one is already an unexpected state worth surfacing loudly
            rather than returning an arbitrary default.
        """
        if not self.paths:
            raise ValueError("Cannot compute bounds of a Drawing with no paths.")
        all_points = np.concatenate([path.points for path in self.paths])
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        return float(min_x), float(min_y), float(max_x), float(max_y)


def _skeleton_neighbor_counts(mask01: np.ndarray) -> np.ndarray:
    """Per-pixel skeleton neighbor count: 1=endpoint, 2=interior, >=3=junction."""
    counts: np.ndarray = (
        cv2.filter2D(mask01, -1, NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT) * mask01
    )
    return counts


def _raw_neighbors(
    y: int, x: int, mask01: np.ndarray, height: int, width: int
) -> list[tuple[int, int]]:
    """Every skeleton-pixel neighbor of ``(y, x)``, regardless of visited status."""
    return [
        (y + dy, x + dx)
        for dy, dx in NEIGHBOR_OFFSETS
        if 0 <= y + dy < height and 0 <= x + dx < width and mask01[y + dy, x + dx]
    ]


def _unvisited_neighbors(
    y: int, x: int, mask01: np.ndarray, consumed: np.ndarray, height: int, width: int
) -> list[tuple[int, int]]:
    """Skeleton-pixel neighbors of ``(y, x)`` not yet marked consumed."""
    return [n for n in _raw_neighbors(y, x, mask01, height, width) if not consumed[n]]


def _walk_from(
    start: tuple[int, int],
    first_step: tuple[int, int],
    *,
    mask01: np.ndarray,
    counts: np.ndarray,
    consumed: np.ndarray,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    """Walk from ``start`` through ``first_step`` to the next junction, endpoint, or dead end.

    Generalizes ``postprocess.prune_short_branches``'s pixel-graph walk:
    steps through interior (degree-2) pixels, marking each consumed, until
    reaching a pixel that isn't degree-2 (an endpoint or a junction, never
    consumed here so other branches can still terminate there) or running
    out of unvisited forward neighbors. ``start`` itself is never marked
    consumed by this function -- callers decide separately whether
    ``start`` (an endpoint or a junction) should be considered consumed.

    Returns
    -------
    list[tuple[int, int]]
        ``(row, col)`` points from ``start`` to wherever the walk stopped,
        inclusive of both ends.
    """
    chain = [start, first_step]
    prev = start
    y, x = first_step
    while counts[y, x] == 2:
        consumed[y, x] = True
        candidates = [
            n for n in _unvisited_neighbors(y, x, mask01, consumed, height, width) if n != prev
        ]
        if len(candidates) != 1:
            break
        prev = (y, x)
        y, x = candidates[0]
        chain.append((y, x))
    return chain


def _trace_cycle(
    start: tuple[int, int], mask01: np.ndarray, consumed: np.ndarray, height: int, width: int
) -> list[tuple[int, int]]:
    """Trace a pure-cycle skeleton component (every pixel degree 2) into one closed chain.

    Only called on components with no endpoint or junction pixels at all
    (e.g. a closed ring outline) -- ``paths_from_mask`` detects those
    up front, since ``postprocess.prune_short_branches``'s endpoint-only
    walk never handles them (a cycle has no endpoint to start from).
    """
    chain = [start]
    consumed[start] = True
    neighbors = _raw_neighbors(*start, mask01, height, width)
    if not neighbors:
        return chain
    prev = start
    y, x = neighbors[0]
    while True:
        consumed[y, x] = True
        chain.append((y, x))
        forward = [n for n in _raw_neighbors(y, x, mask01, height, width) if n != prev]
        if not forward:
            break
        prev = (y, x)
        y, x = forward[0]
        if (y, x) == start:
            break
    return chain


def paths_from_mask(
    mask: np.ndarray, *, kind: Literal["boundary", "detail"] = "boundary"
) -> tuple[Path, ...]:
    """Thin a raw ink mask and trace its skeleton into ordered vector paths.

    Never calls ``cv2.findContours``: tracing a mask's *boundary* is
    exactly the bug ``docs/DIAGNOSIS.md`` diagnoses in the project's
    former ``nms_centerline`` default -- a stroke rendered with any width
    has two edges, so boundary-tracing draws every real stroke twice. This
    function instead thins the mask to a 1px skeleton and walks its graph
    directly, generalizing ``postprocess.prune_short_branches``'s
    endpoint-to-junction walk into full branch extraction (including
    junction-to-junction bridges no endpoint walk reaches, and pure
    cycles with no endpoint at all).

    Short-branch removal is deliberately *not* done here -- that
    decouples vectorization from pruning. Callers filter noise out via
    ``Drawing.filter(lambda p: p.length >= min_length)`` instead.

    Parameters
    ----------
    mask : np.ndarray
        Single-channel image, any shape ``(H, W)``, nonzero = ink. This is
        the engine's *raw* detector output (e.g. Canny's own edge mask),
        not pre-thinned -- thinning happens inside this function.
    kind : {"boundary", "detail"}, optional
        Set on every returned :class:`Path`, by default ``"boundary"``.

    Returns
    -------
    tuple[Path, ...]
        One path per traced skeleton branch or cycle. Chains with fewer
        than 2 points are dropped.
    """
    height, width = mask.shape[:2]
    long_side = max(height, width)

    ink = (mask > 0).astype(np.uint8) * 255
    skeleton = cv2.ximgproc.thinning(ink, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    mask01 = (skeleton > 0).astype(np.uint8)
    if not mask01.any():
        return ()

    counts = _skeleton_neighbor_counts(mask01)
    consumed = np.zeros(mask01.shape, dtype=bool)
    #: (chain, closed) pairs; closed=True only for Phase 1's pure cycles.
    chains: list[tuple[list[tuple[int, int]], bool]] = []

    num_components, comp_labels = cv2.connectedComponents(mask01, connectivity=8)

    # Thinning a real junction (e.g. two strokes crossing) rarely leaves a
    # single clean vertex pixel -- it typically leaves a small cluster of
    # adjacent pixels that each locally classify as a junction (count>=3).
    # Treating each of those individually as its own graph node would trace
    # every short intra-cluster edge as a spurious extra path. Clustering
    # adjacent junction pixels into one "hub" and only tracking edges that
    # leave the cluster (to a non-junction pixel, or to a different hub)
    # avoids that: intra-cluster edges are simply never added to any hub's
    # edge set, so they're never walked at all.
    num_hubs, hub_labels = cv2.connectedComponents((counts >= 3).astype(np.uint8), connectivity=8)
    hub_edges: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {
        hub_id: set() for hub_id in range(1, num_hubs)
    }
    for hub_id in range(1, num_hubs):
        member_ys, member_xs = np.nonzero(hub_labels == hub_id)
        members = set(zip(member_ys.tolist(), member_xs.tolist(), strict=True))
        for my, mx in members:
            for neighbor in _raw_neighbors(my, mx, mask01, height, width):
                if neighbor not in members:
                    hub_edges[hub_id].add(((my, mx), neighbor))

    # Phase 1: pure cycles -- components with no endpoint or junction pixel
    # at all, which the endpoint-seeded walk below never touches.
    for label in range(1, num_components):
        comp_mask = comp_labels == label
        if np.any(counts[comp_mask] != 2):
            continue
        ys, xs = np.nonzero(comp_mask)
        start = (int(ys[0]), int(xs[0]))
        if consumed[start]:
            continue
        chains.append((_trace_cycle(start, mask01, consumed, height, width), True))

    # Phase 2: walk every branch starting from an endpoint.
    endpoint_ys, endpoint_xs = np.nonzero(counts == 1)
    for y, x in zip(endpoint_ys.tolist(), endpoint_xs.tolist(), strict=True):
        start = (int(y), int(x))
        if consumed[start]:
            continue
        first_candidates = _raw_neighbors(*start, mask01, height, width)
        if not first_candidates:
            continue
        chain = _walk_from(
            start,
            first_candidates[0],
            mask01=mask01,
            counts=counts,
            consumed=consumed,
            height=height,
            width=width,
        )
        consumed[start] = True
        end = chain[-1]
        if counts[end] == 1:
            consumed[end] = True
        elif counts[end] >= 3:
            hub_edges[int(hub_labels[end])].discard((end, chain[-2]))
        chains.append((chain, False))

    # Phase 3: any edges left unwalked are hub-to-hub bridges no endpoint
    # walk could reach.
    for edges in hub_edges.values():
        while edges:
            member, neighbor = edges.pop()
            chain = _walk_from(
                member,
                neighbor,
                mask01=mask01,
                counts=counts,
                consumed=consumed,
                height=height,
                width=width,
            )
            end = chain[-1]
            if counts[end] >= 3:
                hub_edges[int(hub_labels[end])].discard((end, chain[-2]))
            chains.append((chain, False))

    paths = []
    for chain, closed in chains:
        if len(chain) < 2:
            continue
        points = np.array([[x, y] for y, x in chain], dtype=np.float32) / long_side
        paths.append(Path(points=points, closed=closed, kind=kind, source_area=None))
    return tuple(paths)


def paths_from_point_chains(
    chains: list[np.ndarray],
    image_shape: tuple[int, int],
    *,
    closed: bool = False,
    kind: Literal["boundary", "detail"] = "boundary",
) -> tuple[Path, ...]:
    """Wrap already-traced point chains as normalized paths, with no raster round-trip.

    For engines that already have exact traced geometry -- EdgeDrawing's
    ``getSegments()`` (:mod:`coloring_page.engines.chained`), or a
    ``cv2.findContours`` result on an already-1px mask
    (:mod:`coloring_page.engines.region`) -- rasterizing that geometry to
    a mask and re-tracing it with :func:`paths_from_mask` would quantize
    precision the source already has and risk ``cv2.ximgproc.thinning``
    altering topology at junctions the source already resolved correctly.
    This function only reformats: reshape, cast, normalize.

    Parameters
    ----------
    chains : list[np.ndarray]
        Each an ``Nx1x2`` or ``Nx2`` array of pixel coordinates in
        ``(x, y)`` order (OpenCV's own convention, matching this module's).
    image_shape : tuple[int, int]
        ``(height, width)`` the chains' pixel coordinates are relative to.
    closed : bool, optional
        Set on every returned :class:`Path`, by default False.
    kind : {"boundary", "detail"}, optional
        Set on every returned :class:`Path`, by default ``"boundary"``.

    Returns
    -------
    tuple[Path, ...]
        Chains with fewer than 2 points are dropped.
    """
    long_side = max(image_shape)
    paths = []
    for chain in chains:
        points = chain.reshape(-1, 2).astype(np.float32) / long_side
        if len(points) < 2:
            continue
        paths.append(Path(points=points, closed=closed, kind=kind, source_area=None))
    return tuple(paths)


def rasterize(drawing: Drawing, *, long_side_px: int, line_thickness: int = 1) -> np.ndarray:
    """Render a ``Drawing`` to a raster canvas -- a stopgap until real SVG/PDF export exists.

    Parameters
    ----------
    drawing : Drawing
        The vector line art to render.
    long_side_px : int
        Long-side size, in pixels, of the output canvas. The other side
        is derived from :attr:`Drawing.aspect_ratio` (see the module
        docstring's reconstruction formula).
    line_thickness : int, optional
        Output stroke width in pixels, by default 1.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, project convention: ``0`` = ink,
        ``255`` = background.
    """
    if drawing.aspect_ratio >= 1:
        out_w, out_h = long_side_px, max(1, round(long_side_px / drawing.aspect_ratio))
    else:
        out_h, out_w = long_side_px, max(1, round(long_side_px * drawing.aspect_ratio))

    canvas = np.full((out_h, out_w), 255, dtype=np.uint8)
    for path in drawing.paths:
        pixel_points = (path.points * long_side_px).round().astype(np.int32)
        cv2.polylines(
            canvas,
            [pixel_points],
            isClosed=path.closed,
            color=0,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    return canvas
