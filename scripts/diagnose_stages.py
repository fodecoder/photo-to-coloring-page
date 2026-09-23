"""Stage-by-stage metrics for the vector pipeline of one conversion style.

Runs a single image through ``style``'s engine and then through the same
post-hoc steps :func:`coloring_page.api.convert_image` applies, stopping
after every stage to measure the ``Drawing`` it holds. The goal is to
locate *which* stage turns continuous contours into short, disconnected
strokes: the final ``validate()`` report only says that the output has
too many dangling endpoints, not where they came from.

Stages reported:

- ``raster_mask``: the exact mask the engine hands to the tracer
  (captured by wrapping ``paths_from_mask``/``paths_from_point_chains``),
  plus skeleton-graph statistics (junction pixels, junction hubs).
- ``traced``: the tracer's raw paths, before any filtering.
- ``engine_len_filter``: the engine's own output (most engines apply an
  internal length filter and/or simplify).
- ``min_stroke_length``, ``area_filter``, ``simplify``: the three steps of
  :func:`coloring_page.profile.apply_detail`, applied one at a time.
- ``render``: the final page raster from :func:`coloring_page.render.to_png`.

Every stage's image is written through a :class:`~coloring_page.pipeline.DebugSink`
into ``--out-dir``, alongside the engine's own debug stages.

Usage
-----
::

    python scripts/diagnose_stages.py path/to/photo.jpg --out-dir diag/canny
    python scripts/diagnose_stages.py path/to/photo.jpg --style chained --out-dir diag/chained
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

import cv2
import numpy as np

from coloring_page.drawing import (
    Drawing,
    _remove_staircase_pixels,
    _skeleton_neighbor_counts,
    rasterize,
)
from coloring_page.engines.registry import get_engine
from coloring_page.page import fit_transform
from coloring_page.pipeline import (
    DEFAULT_WORKING_DIMENSION,
    DebugSink,
    load_image,
    resize_to_max_dimension,
)
from coloring_page.profile import (
    DETAIL_PRESETS,
    RASTER_RESOLUTION_PRESETS,
    DetailLevel,
    Profile,
    _content_area_mm2,
    apply_detail,
)
from coloring_page.render import to_png
from coloring_page.validate import validate

#: Tracer functions an engine module may import by name. Wrapped in the
#: engine's own module namespace (where the engine looks them up), so the
#: original functions in ``coloring_page.drawing`` are never modified.
TRACER_NAMES = ("paths_from_mask", "paths_from_point_chains")


class RecordingDebugSink(DebugSink):
    """A ``DebugSink`` that also keeps every saved image in memory, by stage name.

    The engine's own debug stages are still written to disk exactly as
    ``--debug-dir`` would write them; keeping them in memory as well lets
    this script measure them without re-reading the PNGs.
    """

    def __init__(self, directory: Path) -> None:
        """Create the sink; see :class:`coloring_page.pipeline.DebugSink`."""
        super().__init__(directory)
        self.images: dict[str, np.ndarray] = {}

    def save(self, stage_name: str, image: np.ndarray) -> None:
        """Write ``image`` to disk and remember it under ``stage_name``."""
        super().save(stage_name, image)
        self.images[stage_name] = image


@dataclass
class TracerCall:
    """One captured call to a tracer function: its raster/chain input and its paths."""

    tracer: str
    mask: np.ndarray | None
    image_shape: tuple[int, int]
    paths: tuple[Any, ...]


@contextmanager
def capture_tracer_calls(engine: object) -> Iterator[list[TracerCall]]:
    """Temporarily wrap the tracers the engine's module imported, recording every call.

    Parameters
    ----------
    engine : object
        The engine instance; its class's defining module is the namespace
        patched, because ``from coloring_page.drawing import paths_from_mask``
        binds the name there, not in ``coloring_page.drawing``.

    Yields
    ------
    list[TracerCall]
        Filled in as the engine runs.
    """
    module = sys.modules[type(engine).__module__]
    calls: list[TracerCall] = []
    originals: dict[str, Callable[..., Any]] = {}

    def wrap(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if name == "paths_from_mask":
                mask = np.asarray(args[0])
                calls.append(TracerCall(name, mask, mask.shape[:2], result))
            else:
                shape = tuple(args[1])[:2]
                calls.append(TracerCall(name, None, (int(shape[0]), int(shape[1])), result))
            return result

        return wrapper

    for name in TRACER_NAMES:
        original = getattr(module, name, None)
        if original is not None:
            originals[name] = original
            setattr(module, name, wrap(name, original))
    try:
        yield calls
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def isolated_endpoint_count(drawing: Drawing, tol_norm: float) -> int:
    """Count open-path endpoints with no *other* path within ``tol_norm``.

    ``validate()``'s ``dangling_endpoints`` counts two per open path, so a
    contour split into two paths that still meet reads the same as a real
    gap. This count separates the two: an endpoint touching another path
    (fragmentation) is not isolated; an endpoint in empty space (a visible
    gap) is.

    Parameters
    ----------
    drawing : Drawing
    tol_norm : float
        Match radius, in normalized units.

    Returns
    -------
    int
    """
    if not drawing.paths or tol_norm <= 0:
        return 0
    cell = tol_norm
    grid: dict[tuple[int, int], set[int]] = {}
    for index, path in enumerate(drawing.paths):
        points = path.points
        if path.closed:
            points = np.vstack([points, points[:1]])
        # Densify every segment so a simplified path with sparse vertices
        # still registers along its whole length, not only at its vertices.
        for start, end in zip(points[:-1], points[1:], strict=True):
            steps = max(1, int(np.ceil(np.linalg.norm(end - start) / (cell / 2))))
            for t in np.linspace(0.0, 1.0, steps + 1):
                x, y = start + (end - start) * t
                grid.setdefault((int(x // cell), int(y // cell)), set()).add(index)

    isolated = 0
    for index, path in enumerate(drawing.paths):
        if path.closed:
            continue
        for x, y in (path.points[0], path.points[-1]):
            cx, cy = int(x // cell), int(y // cell)
            neighbors: set[int] = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbors |= grid.get((cx + dx, cy + dy), set())
            if not neighbors - {index}:
                isolated += 1
    return isolated


def drawing_metrics(drawing: Drawing, scale_mm: float, tol_mm: float) -> dict[str, str]:
    """Summarize one vector stage as table cells.

    Parameters
    ----------
    drawing : Drawing
    scale_mm : float
        Millimeters per normalized unit, from :func:`coloring_page.page.fit_transform`.
    tol_mm : float
        Endpoint match radius for :func:`isolated_endpoint_count`.

    Returns
    -------
    dict[str, str]
    """
    paths = drawing.paths
    if not paths:
        return {"paths": "0"}
    lengths_mm = np.array([p.length for p in paths]) * scale_mm
    open_count = sum(1 for p in paths if not p.closed)
    return {
        "paths": str(len(paths)),
        "median_mm": f"{np.median(lengths_mm):.2f}",
        "p90_mm": f"{np.percentile(lengths_mm, 90):.2f}",
        "total_mm": f"{lengths_mm.sum():.0f}",
        "endpoints": str(2 * open_count),
        "isolated_ep": str(isolated_endpoint_count(drawing, tol_mm / scale_mm)),
        "closed_%": f"{100.0 * (len(paths) - open_count) / len(paths):.1f}",
        "pts/path": f"{np.mean([len(p.points) for p in paths]):.1f}",
    }


def mask_metrics(mask: np.ndarray) -> tuple[dict[str, str], np.ndarray]:
    """Skeleton-graph statistics of the raster the tracer received.

    Uses the same thinning, staircase cleanup, and neighbor-count
    classification as :func:`coloring_page.drawing.paths_from_mask`, so
    ``junction_px`` and ``hubs`` are exactly what the tracer sees as graph
    nodes. ``raw_junction_px`` is the same count on the uncleaned
    Zhang-Suen skeleton, for comparison.

    Parameters
    ----------
    mask : np.ndarray
        Tracer input, nonzero = ink.

    Returns
    -------
    tuple[dict[str, str], np.ndarray]
        Table cells, and a BGR visualization: skeleton in black, junction
        pixels in red.
    """
    ink = (mask > 0).astype(np.uint8) * 255
    skeleton = cv2.ximgproc.thinning(ink, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    raw01 = (skeleton > 0).astype(np.uint8)
    raw_junction_px = int((_skeleton_neighbor_counts(raw01) >= 3).sum())
    raw_px = max(1, int(raw01.sum()))
    skel01 = _remove_staircase_pixels(raw01)
    counts = _skeleton_neighbor_counts(skel01)
    junctions = (counts >= 3).astype(np.uint8)
    num_hubs, _ = cv2.connectedComponents(junctions, connectivity=8)
    num_components, _ = cv2.connectedComponents(skel01, connectivity=8)

    vis = np.full((*mask.shape[:2], 3), 255, dtype=np.uint8)
    vis[skel01 > 0] = (0, 0, 0)
    vis[junctions > 0] = (0, 0, 255)
    skeleton_px = int(skel01.sum())
    junction_px = int(junctions.sum())
    cells = {
        "ink_px": str(int((mask > 0).sum())),
        "skeleton_px": str(skeleton_px),
        "raw_junction_px": f"{raw_junction_px} ({100.0 * raw_junction_px / raw_px:.1f}%)",
        "junction_px": f"{junction_px} ({100.0 * junction_px / max(1, skeleton_px):.1f}%)",
        "hubs": str(num_hubs - 1),
        "skel_endpoints": str(int((counts == 1).sum())),
        "components": str(num_components - 1),
    }
    return cells, vis


def render_metrics(page: np.ndarray) -> dict[str, str]:
    """Raster-level connectivity of the final rendered page.

    Parameters
    ----------
    page : np.ndarray
        Output of :func:`coloring_page.render.to_png` (0 = ink).

    Returns
    -------
    dict[str, str]
    """
    ink = (page < 128).astype(np.uint8) * 255
    skeleton = cv2.ximgproc.thinning(ink, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skel01 = (skeleton > 0).astype(np.uint8)
    counts = _skeleton_neighbor_counts(skel01)
    num_components, _ = cv2.connectedComponents(ink, connectivity=8)
    return {
        "ink_px": str(int((ink > 0).sum())),
        "components": str(num_components - 1),
        "skel_endpoints": str(int((counts == 1).sum())),
    }


def paths_visualization(drawing: Drawing, long_side_px: int) -> np.ndarray:
    """Draw each path in its own color with its endpoints marked.

    A single-color render hides fragmentation (adjacent fragments look
    like one line); distinct colors plus endpoint dots make every split
    visible.

    Parameters
    ----------
    drawing : Drawing
    long_side_px : int

    Returns
    -------
    np.ndarray
        BGR image.
    """
    gray = rasterize(drawing, long_side_px=long_side_px)
    canvas = np.full((*gray.shape, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(0)
    for path in drawing.paths:
        color = tuple(int(c) for c in rng.integers(0, 200, size=3))
        pts = (path.points * long_side_px).round().astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=path.closed, color=color, thickness=1)
    for path in drawing.paths:
        if path.closed:
            continue
        for x, y in (path.points[0], path.points[-1]):
            center = (round(float(x) * long_side_px), round(float(y) * long_side_px))
            cv2.circle(canvas, center, 2, (0, 0, 255), -1)
    return canvas


def print_table(rows: list[tuple[str, dict[str, str]]]) -> None:
    """Print ``(stage, cells)`` rows as one markdown table, union of all columns."""
    columns: list[str] = []
    for _, cells in rows:
        columns.extend(c for c in cells if c not in columns)
    print("| stage | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for stage, cells in rows:
        print(f"| {stage} | " + " | ".join(cells.get(c, "") for c in columns) + " |")


def build_parser() -> argparse.ArgumentParser:
    """Construct this script's argument parser."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image", type=Path, help="Input photo (JPG/PNG).")
    parser.add_argument("--style", default="canny", help="Conversion style (default: canny).")
    parser.add_argument(
        "--detail",
        choices=get_args(DetailLevel),
        default=Profile().detail,
        help="Detail level whose apply_detail thresholds are applied (default: Profile's).",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for per-stage PNGs.")
    parser.add_argument(
        "--endpoint-tol-mm",
        type=float,
        default=0.5,
        help="Radius within which an endpoint counts as touching another path (default: 0.5).",
    )
    parser.add_argument("--device", default="auto", help="Inference device for ML styles.")
    return parser


def spatial_region_areas(labels: np.ndarray) -> np.ndarray:
    """Pixel area of every *spatially connected* region of a label image.

    The region engine's labels group pixels by quantized color identity,
    so one label can cover several disjoint patches; a coloring page's
    regions are the connected patches. Each label is labeled separately
    inside its own bounding box, which keeps this fast even with
    thousands of tiny labels.

    Parameters
    ----------
    labels : np.ndarray
        Integer label image, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        1-D array of region areas, in pixels.
    """
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    ends = np.r_[starts[1:], len(flat)]
    width = labels.shape[1]
    areas: list[int] = []
    for start, end in zip(starts, ends, strict=True):
        indices = order[start:end]
        ys, xs = np.divmod(indices, width)
        y0, x0 = int(ys.min()), int(xs.min())
        patch = np.zeros((int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1), dtype=np.uint8)
        patch[ys - y0, xs - x0] = 1
        count, _, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=4)
        areas.extend(int(a) for a in stats[1:count, cv2.CC_STAT_AREA])
    return np.array(areas, dtype=np.int64)


def region_stage_rows(
    engine: Any, image: np.ndarray, drawing: Drawing, threshold_px: float, mm2_per_px: float
) -> list[tuple[str, dict[str, str]]]:
    """Recompute ``RegionEngine``'s internal stages and count what survives each.

    Uses the engine's own stage functions and parameters, so the counts
    are what ``convert`` actually saw rather than a reimplementation.

    Parameters
    ----------
    engine : Any
        A :class:`~coloring_page.engines.region.RegionEngine`.
    image : np.ndarray
        The working-resolution BGR image passed to ``convert``.
    drawing : Drawing
        ``convert``'s output, for the ``source_area`` distribution.
    threshold_px : float
        The detail preset's ``min_region_area_mm2``, converted to pixels.
    mm2_per_px : float
        Printed area of one working-resolution pixel.

    Returns
    -------
    list[tuple[str, dict[str, str]]]
    """
    from coloring_page.engines import region
    from coloring_page.pipeline import derive_kernel_size

    working_px = max(image.shape[:2])
    total_px = image.shape[0] * image.shape[1]
    flattened = region._flatten(image, lambda_=engine.l0_lambda, kappa=engine.l0_kappa)
    if engine.segmentation == "mean_shift":
        spatial_radius = derive_kernel_size(working_px, fraction=0.01, min_value=5, odd=False)
        labels = region.segment_mean_shift(flattened, sp=spatial_radius, sr=engine.color_radius)
    else:
        labels = region.segment_superpixels(
            flattened, algorithm=engine.segmentation, num_superpixels=engine.num_superpixels
        )
    label_areas = np.bincount(labels.ravel())
    spatial_areas = spatial_region_areas(labels)

    lab = cv2.cvtColor(flattened, cv2.COLOR_BGR2LAB)
    if engine.boundary_detection == "labels":
        boundary = region._label_boundaries(labels)
    else:
        boundary = region.gradient_boundaries(lab, percentile=engine.gradient_percentile)
    filtered = region.filter_by_contrast(boundary, labels, lab, min_contrast=engine.min_contrast)

    source_px = np.array(
        [p.source_area * total_px for p in drawing.paths if p.source_area is not None]
    )

    def area_cells(areas: np.ndarray) -> dict[str, str]:
        if len(areas) == 0:
            return {"count": "0"}
        return {
            "count": str(len(areas)),
            "median_px": f"{np.median(areas):.0f}",
            "p90_px": f"{np.percentile(areas, 90):.0f}",
            "median_mm2": f"{np.median(areas) * mm2_per_px:.2f}",
            "above_area_thr": str(int((areas >= threshold_px).sum())),
        }

    return [
        ("segment: color labels", area_cells(label_areas[label_areas > 0])),
        ("segment: spatial regions", area_cells(spatial_areas)),
        ("boundary: raw px", {"count": str(int(boundary.sum()))}),
        ("boundary: after contrast filter px", {"count": str(int(filtered.sum()))}),
        ("engine paths: source_area", area_cells(source_px)),
    ]


def main(argv: list[str] | None = None) -> int:
    """Script entry point."""
    args = build_parser().parse_args(argv)
    profile = Profile(style=args.style, detail=args.detail)
    sink = RecordingDebugSink(args.out_dir)

    image = resize_to_max_dimension(load_image(args.image), DEFAULT_WORKING_DIMENSION)
    working_px = max(image.shape[:2])
    engine = get_engine(
        args.style, device=args.device, resolution=RASTER_RESOLUTION_PRESETS[args.detail]
    )
    with capture_tracer_calls(engine) as calls:
        artwork = engine.convert(image, debug=sink)
    if not isinstance(artwork, Drawing):
        print(f"Style {args.style!r} returns raster output; nothing vector to diagnose.")
        return 1

    scale_mm, _, _ = fit_transform(artwork, profile.page)
    tol = args.endpoint_tol_mm
    rows: list[tuple[str, dict[str, str]]] = []

    # Engines that wrap one contour per call (e.g. region) would otherwise
    # produce one table row per contour: report their chains as one stage.
    chain_calls = [c for c in calls if c.mask is None]
    if len(chain_calls) > 1:
        merged = tuple(p for c in chain_calls for p in c.paths)
        calls = [c for c in calls if c.mask is not None]
        calls.append(
            TracerCall("paths_from_point_chains", None, chain_calls[0].image_shape, merged)
        )

    for i, call in enumerate(calls):
        suffix = f"[{i}]" if len(calls) > 1 else ""
        if call.mask is not None:
            cells, skeleton_vis = mask_metrics(call.mask)
            rows.append((f"raster_mask{suffix}", cells))
            sink.save(f"diag_raster_mask{suffix}", call.mask)
            sink.save(f"diag_skeleton_junctions{suffix}", skeleton_vis)
        traced = Drawing(paths=tuple(call.paths), aspect_ratio=artwork.aspect_ratio)
        rows.append((f"traced ({call.tracer}){suffix}", drawing_metrics(traced, scale_mm, tol)))
        sink.save(f"diag_traced{suffix}", paths_visualization(traced, working_px))

    stage = artwork
    rows.append(("engine_len_filter", drawing_metrics(stage, scale_mm, tol)))
    sink.save("diag_engine_output", paths_visualization(stage, working_px))

    # Mirrors apply_detail step by step; the assertion below keeps this
    # replica honest against the real function.
    params = DETAIL_PRESETS[args.detail]
    min_length_norm = params.min_stroke_length_mm / scale_mm
    stage = stage.filter(lambda p: p.length >= min_length_norm)
    rows.append(
        (
            f"min_stroke_length ({params.min_stroke_length_mm}mm)",
            drawing_metrics(stage, scale_mm, tol),
        )
    )
    sink.save("diag_min_stroke_length", paths_visualization(stage, working_px))

    content_area = _content_area_mm2(scale_mm, artwork.aspect_ratio)
    if content_area > 0:
        min_area_norm = params.min_region_area_mm2 / content_area
        stage = stage.filter(lambda p: p.source_area is None or p.source_area >= min_area_norm)
    rows.append(
        (f"area_filter ({params.min_region_area_mm2}mm2)", drawing_metrics(stage, scale_mm, tol))
    )
    sink.save("diag_area_filter", paths_visualization(stage, working_px))

    stage = stage.simplify(params.simplify_epsilon_mm / scale_mm)
    rows.append(
        (f"simplify ({params.simplify_epsilon_mm}mm)", drawing_metrics(stage, scale_mm, tol))
    )
    sink.save("diag_simplify", paths_visualization(stage, working_px))

    reference = apply_detail(artwork, profile)
    assert len(reference.paths) == len(stage.paths), (
        f"Stepwise replica diverged from apply_detail: {len(stage.paths)} vs "
        f"{len(reference.paths)} paths -- update this script to match profile.apply_detail."
    )

    page = to_png(stage, profile.page)
    render_cells = render_metrics(page)
    report = validate(stage, profile.page)
    render_cells["enclosed_regions"] = str(report.enclosed_regions)
    render_cells["ink_coverage"] = f"{report.ink_coverage:.4f}"
    rows.append(("render (page raster)", render_cells))
    sink.save("diag_render", page)

    print(
        f"style={args.style} detail={args.detail} working_px={working_px} "
        f"scale={scale_mm:.1f} mm/unit ({scale_mm / working_px:.3f} mm/px) "
        f"endpoint_tol={tol}mm\n"
    )
    if args.style == "region":
        mm2_per_px = (scale_mm / working_px) ** 2
        threshold_px = params.min_region_area_mm2 / mm2_per_px
        print(
            f"Region stages (area threshold {params.min_region_area_mm2} mm2 = "
            f"{threshold_px:.0f} px at {mm2_per_px:.4f} mm2/px):\n"
        )
        print_table(region_stage_rows(engine, image, artwork, threshold_px, mm2_per_px))
        print()
    print_table(rows)
    print(f"\nStage images written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
