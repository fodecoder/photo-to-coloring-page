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

import functools
import math
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

#: Pairs of perpendicular 4-neighbors, in ``(dy, dx)`` order. A skeleton
#: pixel with both members of any pair set is a staircase corner: the two
#: neighbors already touch diagonally, so the pixel adds no connectivity.
_ORTHOGONAL_PAIRS = [
    ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)),
    ((1, 0), (0, -1)),
    ((0, -1), (-1, 0)),
]

#: Largest deviation from a straight continuation, in degrees, for two
#: branches meeting at a real junction (3+ branches) to be traced as one
#: path. Generous enough to follow a contour through a junction where a
#: side branch leaves it, strict enough that the two arms of a T are
#: never mistaken for one bent stroke.
MAX_CONTINUATION_DEVIATION_DEG = 45.0

#: A dead-end branch leaving a junction is a thinning artifact (a "spur")
#: rather than a real stroke when it is no longer than
#: ``SPUR_RADIUS_FACTOR * r + SPUR_EXTRA_PX`` pixels, where ``r`` is the
#: ink's local half-width at that junction. Zhang-Suen grows such spurs
#: toward the corners of any stroke wider than 1px -- most visibly at an
#: acute corner, where a spur would otherwise turn a closed outline into
#: a 3-way junction. Scaling with ``r`` keeps the threshold meaningful
#: for both 1px Canny edges and thick ML-engine strokes.
SPUR_RADIUS_FACTOR = 2.0
SPUR_EXTRA_PX = 2.0


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


@functools.cache
def _simple_point_lut() -> np.ndarray:
    """Whether a skeleton pixel is a *simple point*, indexed by its 8-neighbor bit code.

    A simple point can be deleted without changing the skeleton's
    topology: its foreground neighbors form exactly one 8-connected
    component (so no stroke is cut), and its background neighbors that
    are 4-adjacent to it form exactly one 4-connected component (so no
    enclosed hole is opened or created). Bit ``i`` of the code is set when
    the neighbor at ``NEIGHBOR_OFFSETS[i]`` is ink.

    Returns
    -------
    np.ndarray
        Shape ``(256,)``, dtype ``bool``.
    """
    lut = np.zeros(256, dtype=bool)
    orthogonal = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for code in range(256):
        patch = np.zeros((3, 3), dtype=np.uint8)
        for bit, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
            if code >> bit & 1:
                patch[dy + 1, dx + 1] = 1
        num_fg, _ = cv2.connectedComponents(patch, connectivity=8)
        background = (1 - patch).astype(np.uint8)
        background[1, 1] = 0
        _, bg_labels = cv2.connectedComponents(background, connectivity=4)
        bg_components = {int(bg_labels[y, x]) for y, x in orthogonal if background[y, x]}
        lut[code] = num_fg - 1 == 1 and len(bg_components) == 1
    return lut


def _remove_staircase_pixels(mask01: np.ndarray) -> np.ndarray:
    """Reduce a Zhang-Suen skeleton to a minimal 8-connected one.

    Zhang-Suen thinning leaves L-shaped "staircase" corners wherever a
    stroke wider than 1px runs diagonally or curves: a pixel whose
    horizontal and vertical neighbors are both ink, even though those two
    already touch diagonally. Each such pixel has 3+ skeleton neighbors,
    so it is indistinguishable from a real junction by neighbor count --
    on a real photo nearly half of all skeleton pixels were classified as
    junctions this way, which is what shattered every curved contour into
    short straight fragments. This removes those corners, keeping only
    simple points (see :func:`_simple_point_lut`) so topology is preserved.

    Parameters
    ----------
    mask01 : np.ndarray
        ``(H, W)`` ``uint8`` skeleton, ``1`` = skeleton pixel.

    Returns
    -------
    np.ndarray
        A new ``(H, W)`` ``uint8`` array, ``1`` = skeleton pixel.
    """
    padded = np.pad(mask01, 1).astype(np.uint8)
    north, south = padded[:-2, 1:-1], padded[2:, 1:-1]
    west, east = padded[1:-1, :-2], padded[1:-1, 2:]
    candidates = mask01.astype(bool) & (
        ((north & east) | (east & south) | (south & west) | (west & north)).astype(bool)
    )
    lut = _simple_point_lut()
    # Removal only ever deletes neighbors, so it can't create a new
    # candidate: one sequential pass over the initial candidates suffices,
    # as long as each is re-checked against the *current* state.
    for y, x in zip(*np.nonzero(candidates), strict=True):
        py, px = int(y) + 1, int(x) + 1
        if not any(
            padded[py + ay, px + ax] and padded[py + by, px + bx]
            for (ay, ax), (by, bx) in _ORTHOGONAL_PAIRS
        ):
            continue
        code = 0
        for bit, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
            if padded[py + dy, px + dx]:
                code |= 1 << bit
        if lut[code]:
            padded[py, px] = 0
    return padded[1:-1, 1:-1].copy()


@dataclass
class _Edge:
    """One skeleton-graph edge: a pixel chain between two graph nodes.

    ``start``/``end`` are hub labels (``>= 1``) for a junction cluster,
    or ``0`` for a dead end (a skeleton endpoint, or a walk that ran out
    of unvisited pixels).
    """

    pixels: list[tuple[int, int]]
    start: int
    end: int

    def length(self) -> float:
        """Euclidean length of the pixel chain, in pixels."""
        steps = np.diff(np.asarray(self.pixels, dtype=np.float64), axis=0)
        return float(np.linalg.norm(steps, axis=1).sum())


def _prune_spurs(edges: list[_Edge], spur_threshold: dict[int, float]) -> list[_Edge]:
    """Drop thinning-artifact spurs at junctions, never lowering a hub below 2 branches.

    A spur is a hub-to-dead-end edge (or a tiny hub self-loop) no longer
    than its hub's threshold. Pruning stops at a hub once it would leave
    fewer than 2 branches there: at an acute corner, removing the spur
    leaves the two real sides to be joined through the hub, while
    removing them too would erase real geometry.

    Parameters
    ----------
    edges : list[_Edge]
    spur_threshold : dict[int, float]
        Maximum spur length, in pixels, per hub label.

    Returns
    -------
    list[_Edge]
    """
    degree: dict[int, int] = {}
    for edge in edges:
        for node in (edge.start, edge.end):
            if node:
                degree[node] = degree.get(node, 0) + 1

    spurs: list[tuple[float, int, int]] = []
    for index, edge in enumerate(edges):
        if edge.start and edge.start == edge.end:
            hub, removed_degree = edge.start, 2
        elif bool(edge.start) != bool(edge.end):
            hub, removed_degree = edge.start or edge.end, 1
        else:
            continue
        length = edge.length()
        if length <= spur_threshold[hub]:
            spurs.append((length, index, removed_degree))

    dropped: set[int] = set()
    for _, index, removed_degree in sorted(spurs):
        edge = edges[index]
        hub = edge.start or edge.end
        if degree[hub] < 3 or degree[hub] - removed_degree < 2:
            continue
        degree[hub] -= removed_degree
        dropped.add(index)
    return [edge for index, edge in enumerate(edges) if index not in dropped]


def _outgoing_direction(pixels: list[tuple[int, int]], lookahead: int) -> np.ndarray:
    """Unit vector from ``pixels[0]`` toward the pixel ``lookahead`` steps along the chain."""
    target = pixels[min(lookahead, len(pixels) - 1)]
    vector = np.array(target, dtype=np.float64) - np.array(pixels[0], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _pair_edge_ends(
    edges: list[_Edge], lookahead: dict[int, int]
) -> dict[tuple[int, int], tuple[int, int]]:
    """Decide which edge ends continue into each other through each hub.

    An edge end is ``(edge_index, side)``, side ``0`` = ``pixels[0]``,
    ``1`` = ``pixels[-1]``. A hub with exactly 2 ends always joins them,
    at any angle -- that is an ordinary contour passing through (an acute
    corner, or a leftover false junction). A hub with 3+ ends joins the
    most nearly opposite pairs, greedily, while their deviation from a
    straight line is within :data:`MAX_CONTINUATION_DEVIATION_DEG`; any
    end left unpaired terminates at the hub, touching the path that
    continues through it.

    Parameters
    ----------
    edges : list[_Edge]
    lookahead : dict[int, int]
        Per hub label, how many pixels along each edge to look when
        estimating its direction -- far enough to see past the hub's own
        pixel cluster.

    Returns
    -------
    dict[tuple[int, int], tuple[int, int]]
        Symmetric mapping between paired ends.
    """
    ends_at: dict[int, list[tuple[int, int]]] = {}
    for index, edge in enumerate(edges):
        if edge.start:
            ends_at.setdefault(edge.start, []).append((index, 0))
        if edge.end:
            ends_at.setdefault(edge.end, []).append((index, 1))

    max_cos = -math.cos(math.radians(MAX_CONTINUATION_DEVIATION_DEG))
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    for hub, ends in ends_at.items():
        if len(ends) < 2:
            continue
        if len(ends) == 2:
            first, second = ends
            partner[first] = second
            partner[second] = first
            continue

        directions = []
        for index, side in ends:
            pixels = edges[index].pixels
            directions.append(
                _outgoing_direction(pixels if side == 0 else pixels[::-1], lookahead[hub])
            )
        candidates = sorted(
            (float(directions[i] @ directions[j]), i, j)
            for i in range(len(ends))
            for j in range(i + 1, len(ends))
        )
        for cosine, i, j in candidates:
            if cosine > max_cos:
                break
            if ends[i] in partner or ends[j] in partner:
                continue
            partner[ends[i]] = ends[j]
            partner[ends[j]] = ends[i]
    return partner


def _link_edges(
    edges: list[_Edge], partner: dict[tuple[int, int], tuple[int, int]]
) -> list[tuple[list[tuple[int, int]], bool]]:
    """Concatenate paired edges into maximal chains.

    Parameters
    ----------
    edges : list[_Edge]
    partner : dict[tuple[int, int], tuple[int, int]]
        As returned by :func:`_pair_edge_ends`.

    Returns
    -------
    list[tuple[list[tuple[int, int]], bool]]
        ``(chain, closed)`` pairs, chain in ``(row, col)`` pixels.
    """
    visited: set[int] = set()

    def follow(index: int, side: int) -> tuple[list[tuple[int, int]], bool]:
        chain: list[tuple[int, int]] = []
        start = index
        while True:
            visited.add(index)
            pixels = edges[index].pixels
            sequence = pixels if side == 0 else pixels[::-1]
            if chain and chain[-1] == sequence[0]:
                sequence = sequence[1:]
            chain.extend(sequence)
            next_end = partner.get((index, 1 - side))
            if next_end is None:
                return chain, False
            index, side = next_end
            if index == start:
                return chain, True
            if index in visited:
                return chain, False

    chains = []
    # Open chains first: start from every end nothing continues into.
    for index in range(len(edges)):
        for side in (0, 1):
            if index not in visited and (index, side) not in partner:
                chains.append(follow(index, side))
    # Whatever is left has every end paired: closed loops through hubs.
    for index in range(len(edges)):
        if index not in visited:
            chains.append(follow(index, 0))
    return chains


def _drop_collinear_points(points: np.ndarray, *, closed: bool) -> np.ndarray:
    """Remove duplicate and exactly-collinear interior points from an integer pixel chain.

    A traced pixel chain has one vertex per pixel, so a straight run is
    stored as many redundant collinear points, and joining edges through
    a hub leaves its join pixel as one more. Dropping points that lie
    exactly on the straight line between their neighbors (and continue
    in the same direction) is lossless -- length and rendered geometry
    are unchanged -- and leaves only the chain's real turning points.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 2)`` integer pixel coordinates.
    closed : bool
        Whether the chain wraps around (its first point also counts as
        an interior point).

    Returns
    -------
    np.ndarray
        ``(M, 2)``, ``M <= N``; unchanged if compressing would leave a
        closed chain with fewer than 3 points.
    """
    if len(points) < 3:
        return points
    keep_distinct = np.any(points != np.roll(points, 1, axis=0), axis=1)
    if not closed:
        keep_distinct[0] = True
    deduplicated = points[keep_distinct]
    if len(deduplicated) < 3:
        return deduplicated if not closed else points

    incoming = deduplicated - np.roll(deduplicated, 1, axis=0)
    outgoing = np.roll(deduplicated, -1, axis=0) - deduplicated
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = (incoming * outgoing).sum(axis=1)
    keep = (cross != 0) | (dot <= 0)
    if not closed:
        keep[0] = keep[-1] = True
    compressed: np.ndarray = deduplicated[keep]
    if closed and len(compressed) < 3:
        return deduplicated
    return compressed


def paths_from_mask(
    mask: np.ndarray, *, kind: Literal["boundary", "detail"] = "boundary"
) -> tuple[Path, ...]:
    """Thin a raw ink mask and trace its skeleton into ordered vector paths.

    Never calls ``cv2.findContours``: tracing a mask's *boundary* is
    exactly the bug ``docs/DIAGNOSIS.md`` diagnoses in the project's
    former ``nms_centerline`` default -- a stroke rendered with any width
    has two edges, so boundary-tracing draws every real stroke twice. This
    function instead thins the mask to a 1px skeleton and traces its
    graph:

    1. Zhang-Suen thinning, then removal of its staircase corners
       (:func:`_remove_staircase_pixels`), so only real junctions are
       classified as junctions.
    2. The skeleton is split into edges between junction clusters
       ("hubs") and dead ends; pure cycles become closed paths directly.
    3. Thinning spurs at junctions are dropped (:func:`_prune_spurs`).
    4. Edges are joined *through* hubs (:func:`_pair_edge_ends`), so a
       contour stays one path across a junction instead of being split
       there, and a chain of joined edges that returns to its start
       becomes a closed path.

    Short-*stroke* removal is deliberately still not done here -- the
    spur pruning above only removes thinning artifacts, sized by the
    local stroke width. Callers filter noise out via
    ``Drawing.filter(lambda p: p.length >= min_length)``.

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
        One path per traced contour, with exactly-collinear interior
        points removed (see :func:`_drop_collinear_points`). Chains with
        fewer than 2 points are dropped.
    """
    height, width = mask.shape[:2]
    long_side = max(height, width)

    ink = (mask > 0).astype(np.uint8) * 255
    skeleton = cv2.ximgproc.thinning(ink, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    mask01 = _remove_staircase_pixels((skeleton > 0).astype(np.uint8))
    if not mask01.any():
        return ()

    counts = _skeleton_neighbor_counts(mask01)
    consumed = np.zeros(mask01.shape, dtype=bool)
    #: (chain, closed) pairs, in (row, col) pixels.
    chains: list[tuple[list[tuple[int, int]], bool]] = []
    edges: list[_Edge] = []

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
    # Local stroke half-width at each hub, from the *unthinned* ink: sizes
    # both the spur threshold and how far to look along a branch when
    # estimating its direction, so both scale with the stroke's width.
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    hub_radius: dict[int, float] = {}
    for hub_id in range(1, num_hubs):
        member_ys, member_xs = np.nonzero(hub_labels == hub_id)
        hub_radius[hub_id] = float(distance[member_ys, member_xs].max())
        members = set(zip(member_ys.tolist(), member_xs.tolist(), strict=True))
        for my, mx in members:
            for neighbor in _raw_neighbors(my, mx, mask01, height, width):
                if neighbor not in members:
                    hub_edges[hub_id].add(((my, mx), neighbor))

    def node_of(pixel: tuple[int, int]) -> int:
        return int(hub_labels[pixel]) if counts[pixel] >= 3 else 0

    # Pure cycles -- components with no endpoint or junction pixel at
    # all, which the endpoint-seeded walk below never touches.
    for label in range(1, num_components):
        comp_mask = comp_labels == label
        if np.any(counts[comp_mask] != 2):
            continue
        ys, xs = np.nonzero(comp_mask)
        start = (int(ys[0]), int(xs[0]))
        if consumed[start]:
            continue
        chains.append((_trace_cycle(start, mask01, consumed, height, width), True))

    # Edges starting from an endpoint.
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
        edges.append(_Edge(chain, 0, node_of(end)))

    # Any edges left unwalked are hub-to-hub bridges no endpoint walk
    # could reach.
    for hub_id, hub_edge_set in hub_edges.items():
        while hub_edge_set:
            member, neighbor = hub_edge_set.pop()
            # Several members of one hub can touch the same neighbor; once
            # any walk has passed through that neighbor, the remaining
            # member->neighbor edges are duplicates of an edge already
            # traced, not new branches -- walking them would emit
            # spurious 2-pixel stubs.
            if consumed[neighbor]:
                continue
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
            edges.append(_Edge(chain, hub_id, node_of(end)))

    spur_threshold = {
        hub_id: SPUR_RADIUS_FACTOR * radius + SPUR_EXTRA_PX for hub_id, radius in hub_radius.items()
    }
    lookahead = {hub_id: max(3, math.ceil(2 * radius + 2)) for hub_id, radius in hub_radius.items()}
    edges = _prune_spurs(edges, spur_threshold)
    chains.extend(_link_edges(edges, _pair_edge_ends(edges, lookahead)))

    paths = []
    for chain, closed in chains:
        if closed and len(chain) > 1 and chain[-1] == chain[0]:
            chain = chain[:-1]
        if len(chain) < 2:
            continue
        xy = np.array([[x, y] for y, x in chain], dtype=np.int64)
        xy = _drop_collinear_points(xy, closed=closed)
        points = xy.astype(np.float32) / long_side
        paths.append(Path(points=points, closed=closed and len(points) >= 3, kind=kind))
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
