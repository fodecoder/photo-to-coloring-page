"""Spike: does a SAM 2 region partition beat the current gradient-based engines?

Throwaway validation code -- NOT part of ``src/``, NOT covered by ``tests/``,
safe to delete once the PASS/FAIL decision below has been made and written up.

Every engine registered in ``coloring_page.engines.registry`` answers "where
is there a local gradient / a generic line?" (see ``docs/DIAGNOSIS.md``).
That produces unusable output: broken strokes, unclosed contours, background
texture read as noise. This script tests a structurally different approach
before committing to a pipeline rewrite: segment the image into regions with
SAM 2's automatic mask generator, resolve the resulting masks into a single
label map with no holes and no overlaps, discard/merge regions too small to
color, and draw only the boundaries *between* regions -- which are closed by
construction, unlike a gradient threshold's ragged mask.

Decision is made on three numbers only -- ink coverage, region count, and
dangling skeleton endpoints -- deliberately NOT on
``metrics.boundary_f_measure`` (see ``scripts/metrics.py``), which
``docs/DIAGNOSIS.md`` §4 shows rewards stroke thickness and recall against a
hand-drawn reference, not colorability.

Requires the optional ``seg`` extra (``pip install -e ".[seg]"``) and the
SAM 2.1 Hiera-small checkpoint (``python scripts/fetch_weights.py --filename
sam2.1_hiera_small.pt``).

Usage
-----
::

    python scripts/spike_segmentation.py --limit 1
    python scripts/spike_segmentation.py --limit 3
    python scripts/spike_segmentation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# scripts/ is not a package; this script is invoked directly
# (`python scripts/spike_segmentation.py`), which puts its own directory --
# not the repo root -- at the front of sys.path, so a plain module import
# reaches compare.py without needing an __init__.py.
from compare import REFERENCE_PAIRS, _label_tile  # noqa: E402
from scipy import ndimage

from coloring_page.drawing import rasterize
from coloring_page.engines.registry import get_engine
from coloring_page.pipeline import load_image, run_pipeline
from coloring_page.postprocess import redraw_segments
from coloring_page.validate import ink_coverage

#: 8-neighborhood sum kernel (excludes the center pixel itself). Duplicated
#: from coloring_page.postprocess's private _NEIGHBOR_KERNEL rather than
#: imported, since this script must not require changes to src/.
_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

#: 4-connected neighbor offsets used to find where region labels change.
_NEIGHBOR_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

#: Sentinel for a not-yet-assigned pixel in the label map under construction.
_UNASSIGNED = -1

#: Exit-criterion thresholds (see module docstring; NOT boundary_f_measure).
_MIN_INK_COVERAGE = 0.03
_MAX_INK_COVERAGE = 0.08
_MIN_REGION_COUNT = 40
_MAX_REGION_COUNT = 400


def load_mask_generator(checkpoint: Path, *, device: str) -> Any:
    """Build a SAM 2.1 Hiera-small automatic mask generator.

    Parameters
    ----------
    checkpoint : Path
        Path to the ``sam2.1_hiera_small.pt`` checkpoint, as fetched by
        ``scripts/fetch_weights.py``.
    device : str
        ``"cuda"`` or ``"cpu"``, passed straight to ``sam2.build_sam.build_sam2``.

    Returns
    -------
    Any
        A ``sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator``,
        typed ``Any`` since ``sam2`` is an optional dependency not present
        in this project's default type-checking environment.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_s.yaml",
        ckpt_path=str(checkpoint),
        device=device,
    )
    return SAM2AutomaticMaskGenerator(model)


def generate_masks(mask_generator: Any, image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """Run SAM 2's automatic mask generator over a BGR image.

    Parameters
    ----------
    mask_generator : Any
        As returned by :func:`load_mask_generator`.
    image_bgr : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

    Returns
    -------
    list[dict[str, Any]]
        One dict per proposed mask, each with (among other keys) a boolean
        ``"segmentation"`` array of shape ``(H, W)`` and a numeric ``"area"``.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result: list[dict[str, Any]] = mask_generator.generate(image_rgb)
    return result


def _mean_lab_per_label(lab_image: np.ndarray, labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Mean Lab color of each label.

    Local copy of ``coloring_page.engines.region``'s private helper of the
    same name -- duplicated rather than imported since it isn't part of
    that module's public surface and this script must not require ``src/``
    changes.

    Parameters
    ----------
    lab_image : np.ndarray
        Image in Lab color space, shape ``(H, W, 3)``.
    labels : np.ndarray
        Integer label image, shape ``(H, W)``, values in ``[0, num_labels)``.
    num_labels : int
        Number of distinct labels.

    Returns
    -------
    np.ndarray
        Shape ``(num_labels, 3)``, mean Lab color per label.
    """
    flat_lab = lab_image.reshape(-1, 3).astype(np.float64)
    flat_labels = labels.reshape(-1)
    sums = np.zeros((num_labels, 3), dtype=np.float64)
    counts = np.zeros(num_labels, dtype=np.float64)
    np.add.at(sums, flat_labels, flat_lab)
    np.add.at(counts, flat_labels, 1)
    return sums / np.maximum(counts, 1)[:, None]


def build_label_map(masks: list[dict[str, Any]], image_shape: tuple[int, int]) -> np.ndarray:
    """Resolve overlapping SAM masks into one complete, non-overlapping label map.

    Each pixel is assigned to the *smallest* mask that contains it: masks
    are painted smallest-area-first, so a later (larger) mask only fills
    pixels no smaller mask already claimed. Pixels no mask covers at all
    are filled from their nearest already-assigned pixel (spatial nearest-
    neighbor via a distance transform) -- for a small gap, the spatially
    nearest pixel's region is also the best Lab match, so this single pass
    satisfies both readings of "nearest region, most similar in Lab."

    Parameters
    ----------
    masks : list[dict[str, Any]]
        As returned by :func:`generate_masks`.
    image_shape : tuple[int, int]
        ``(height, width)`` of the source image.

    Returns
    -------
    np.ndarray
        Integer label image, shape ``image_shape``, dtype ``int32``, with
        every pixel assigned to some label and no holes.
    """
    labels = np.full(image_shape, _UNASSIGNED, dtype=np.int32)
    for new_id, mask in enumerate(sorted(masks, key=lambda m: m["area"])):
        segmentation = mask["segmentation"]
        labels[segmentation & (labels == _UNASSIGNED)] = new_id

    unassigned = labels == _UNASSIGNED
    if unassigned.any():
        if unassigned.all():
            raise ValueError("SAM produced no masks covering any pixel of this image.")
        _, indices = ndimage.distance_transform_edt(unassigned, return_indices=True)
        nearest_labels = labels[tuple(indices)]
        labels[unassigned] = nearest_labels[unassigned]

    return labels


def _label_adjacency(labels: np.ndarray) -> set[tuple[int, int]]:
    """Unique unordered pairs of labels that are 4-adjacent somewhere in ``labels``."""
    pairs: set[tuple[int, int]] = set()
    height, width = labels.shape
    for dy, dx in _NEIGHBOR_OFFSETS:
        y0, y1 = max(0, -dy), height - max(0, dy)
        x0, x1 = max(0, -dx), width - max(0, dx)
        a = labels[y0:y1, x0:x1]
        b = labels[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        differing = a != b
        for label_a, label_b in zip(a[differing].tolist(), b[differing].tolist(), strict=True):
            pairs.add((min(label_a, label_b), max(label_a, label_b)))
    return pairs


def merge_small_regions(
    labels: np.ndarray,
    lab_image: np.ndarray,
    *,
    min_area_fraction: float = 0.0015,
    max_iterations: int = 20,
) -> np.ndarray:
    """Merge regions smaller than ``min_area_fraction`` into their closest-Lab neighbor.

    Iterates to convergence: each pass recomputes areas, per-label mean Lab
    colors, and adjacency (merging changes all three), then merges every
    still-too-small region into whichever adjacent region has the smallest
    Lab Euclidean distance to it. Stops when either no region remains below
    the area threshold, or a full pass makes zero merges (guards against a
    pathological cycle where merging one tiny region creates another one
    just below threshold), capped at ``max_iterations``.

    Parameters
    ----------
    labels : np.ndarray
        Integer label image, shape ``(H, W)``, as returned by
        :func:`build_label_map`. Need not have contiguous label ids.
    lab_image : np.ndarray
        Image in Lab color space, same shape as ``labels`` plus a channel axis.
    min_area_fraction : float, optional
        Regions smaller than this fraction of the image area are merged
        away, by default 0.0015 (0.15%).
    max_iterations : int, optional
        Safety cap on merge passes, by default 20.

    Returns
    -------
    np.ndarray
        Integer label image, same shape as ``labels``, dtype ``int32``,
        relabeled to contiguous ids ``0..N-1``.
    """
    height, width = labels.shape
    min_area = min_area_fraction * height * width

    for _iteration in range(max_iterations):
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.reshape(height, width).astype(np.int32)
        num_labels = int(labels.max()) + 1

        areas = np.bincount(labels.ravel(), minlength=num_labels)
        too_small = np.nonzero(areas < min_area)[0]
        if len(too_small) == 0:
            break

        mean_lab = _mean_lab_per_label(lab_image, labels, num_labels)
        adjacency = _label_adjacency(labels)
        neighbors_of: dict[int, list[int]] = {label: [] for label in range(num_labels)}
        for a, b in adjacency:
            neighbors_of[a].append(b)
            neighbors_of[b].append(a)

        remap = np.arange(num_labels)
        merges = 0
        for label in too_small.tolist():
            candidates = [n for n in neighbors_of.get(label, []) if n != label]
            if not candidates:
                continue
            distances = [
                float(np.linalg.norm(mean_lab[label] - mean_lab[candidate]))
                for candidate in candidates
            ]
            best = candidates[int(np.argmin(distances))]
            remap[label] = best
            merges += 1

        if merges == 0:
            break
        labels = remap[labels]
    else:
        print(
            f"merge_small_regions: hit max_iterations={max_iterations} without full "
            "convergence; proceeding with the current label map.",
            file=sys.stderr,
        )

    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(height, width).astype(np.int32)


def draw_label_boundaries(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Trace each label's boundary with ``cv2.findContours`` and redraw at uniform width.

    Uses ``cv2.RETR_EXTERNAL`` per label rather than a boundary-mask
    approach: a boundary shared by two labels is traced once, as the
    external contour of each of them in turn, so ``RETR_LIST``/``RETR_CCOMP``
    here would double-draw every shared edge.

    Parameters
    ----------
    labels : np.ndarray
        Integer label image, shape ``(H, W)``, with contiguous ids.
    shape : tuple[int, int]
        ``(height, width)`` of the output canvas.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, project convention: ``0`` = ink,
        ``255`` = background.
    """
    segments: list[np.ndarray] = []
    for label_id in range(int(labels.max()) + 1):
        mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        segments.extend(contours)
    return redraw_segments(segments, shape, polyline_epsilon=1.2, line_thickness=1)


def count_dangling_endpoints(line_art: np.ndarray) -> int:
    """Count skeleton pixels with exactly one neighbor in ``line_art``.

    Parameters
    ----------
    line_art : np.ndarray
        Single-channel ``uint8`` image, project convention: ``0`` = ink,
        ``255`` = background.

    Returns
    -------
    int
        Number of dangling endpoints. A fully closed set of boundaries
        (as a label map's boundaries are, by construction) should have none;
        a nonzero count here indicates a bug upstream, e.g. in
        :func:`build_label_map`.
    """
    mask = (line_art < 128).astype(np.uint8)
    if not mask.any():
        return 0
    skeleton = cv2.ximgproc.thinning(mask * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    binary_skeleton = (skeleton > 0).astype(np.uint8)
    counts = (
        cv2.filter2D(binary_skeleton, -1, _NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT)
        * binary_skeleton
    )
    return int(np.count_nonzero((binary_skeleton == 1) & (counts == 1)))


def region_count(labels: np.ndarray) -> int:
    """Number of distinct labels in ``labels``."""
    return int(len(np.unique(labels)))


def decide_pass_fail(metrics: dict[str, Any]) -> bool:
    """Apply the three exit-criterion thresholds to one image's metrics.

    Parameters
    ----------
    metrics : dict[str, Any]
        Must contain ``"ink_coverage"``, ``"dangling_endpoints"``, and
        ``"region_count"``, as produced by :func:`evaluate_image`.

    Returns
    -------
    bool
        True if all three criteria pass.
    """
    return (
        _MIN_INK_COVERAGE <= metrics["ink_coverage"] <= _MAX_INK_COVERAGE
        and metrics["dangling_endpoints"] == 0
        and _MIN_REGION_COUNT <= metrics["region_count"] <= _MAX_REGION_COUNT
    )


def evaluate_image(
    image_path: Path, mask_generator: Any, *, min_area_fraction: float
) -> dict[str, Any]:
    """Run the full SAM partition pipeline on one image and compute its metrics.

    Parameters
    ----------
    image_path : Path
        Path to a photo under ``docs/``.
    mask_generator : Any
        As returned by :func:`load_mask_generator`.
    min_area_fraction : float
        Passed to :func:`merge_small_regions`.

    Returns
    -------
    dict[str, Any]
        ``image``, ``region_count``, ``ink_coverage``, ``dangling_endpoints``,
        ``passed``, and ``line_art`` (the rendered boundary image, for the
        contact sheet).
    """
    image = load_image(image_path)
    masks = generate_masks(mask_generator, image)
    labels = build_label_map(masks, image.shape[:2])
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    labels = merge_small_regions(labels, lab_image, min_area_fraction=min_area_fraction)
    line_art = draw_label_boundaries(labels, image.shape[:2])

    metrics: dict[str, Any] = {
        "image": image_path.name,
        "region_count": region_count(labels),
        "ink_coverage": ink_coverage(line_art),
        "dangling_endpoints": count_dangling_endpoints(line_art),
        "line_art": line_art,
    }
    metrics["passed"] = decide_pass_fail(metrics)
    return metrics


def build_contact_sheet(
    input_dir: Path, ref_dir: Path, mask_generator: Any, *, min_area_fraction: float
) -> np.ndarray:
    """Build a 4-column contact sheet: input | desired | region engine | SAM partition.

    One row per image with a known reference (see ``compare.REFERENCE_PAIRS``).

    Parameters
    ----------
    input_dir : Path
        Directory containing ``starting-image*.jpeg``.
    ref_dir : Path
        Directory containing ``desired*.jpg`` reference drawings.
    mask_generator : Any
        As returned by :func:`load_mask_generator`.
    min_area_fraction : float
        Passed to :func:`merge_small_regions`.

    Returns
    -------
    np.ndarray
        A single BGR contact-sheet image.
    """
    region_engine = get_engine("region")
    rows = []
    for input_name, ref_name in REFERENCE_PAIRS.items():
        input_path = input_dir / input_name
        if not input_path.exists():
            print(f"Skipping contact-sheet row for missing {input_path}", file=sys.stderr)
            continue

        image = load_image(input_path)
        desired = cv2.imread(str(ref_dir / ref_name), cv2.IMREAD_GRAYSCALE)
        region_drawing = run_pipeline(image, region_engine)
        region_result = rasterize(region_drawing, long_side_px=max(image.shape[:2]))
        sam_metrics = evaluate_image(
            input_path, mask_generator, min_area_fraction=min_area_fraction
        )

        tiles = [
            _label_tile(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), "input"),
            _label_tile(desired, "desired") if desired is not None else None,
            _label_tile(region_result, "region (current)"),
            _label_tile(sam_metrics["line_art"], "SAM partition"),
        ]
        tiles = [tile for tile in tiles if tile is not None]

        max_height = max(tile.shape[0] for tile in tiles)
        padded = [
            cv2.copyMakeBorder(
                tile,
                0,
                max_height - tile.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            for tile in tiles
        ]
        rows.append(cv2.hconcat(padded))

    if not rows:
        raise ValueError("No referenced images found; cannot build a contact sheet.")

    max_width = max(row.shape[1] for row in rows)
    padded_rows = [
        cv2.copyMakeBorder(
            row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        for row in rows
    ]
    return cv2.vconcat(padded_rows)


def print_report(results: list[dict[str, Any]]) -> bool:
    """Print a per-image metrics line and a final overall PASS/FAIL line.

    Parameters
    ----------
    results : list[dict[str, Any]]
        As returned by repeated calls to :func:`evaluate_image`.

    Returns
    -------
    bool
        Overall pass/fail: True only if every image passed.
    """
    for result in results:
        verdict = "PASS" if result["passed"] else "FAIL"
        print(
            f"{result['image']}: regions={result['region_count']}, "
            f"ink_coverage={result['ink_coverage']:.4f}, "
            f"dangling_endpoints={result['dangling_endpoints']} -> {verdict}"
        )
    overall = all(result["passed"] for result in results)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'} ({len(results)} image(s))")
    return overall


def _iter_input_images(input_dir: Path, *, limit: int | None) -> list[Path]:
    """Sorted ``starting-image*.jpeg`` files in ``input_dir``, optionally capped at ``limit``."""
    paths = sorted(input_dir.glob("starting-image*.jpeg"))
    return paths if limit is None else paths[:limit]


def build_parser() -> argparse.ArgumentParser:
    """Construct this script's argument parser."""
    parser = argparse.ArgumentParser(
        description="Validate a SAM 2 region partition against this project's gradient-based "
        "engines (see the module docstring for the exit criteria)."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("docs"), help="Directory of starting-image*.jpeg."
    )
    parser.add_argument(
        "--ref-dir", type=Path, default=Path("docs"), help="Directory of desired*.jpg references."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / ".cache" / "coloring_page" / "sam2.1_hiera_small.pt",
        help="Path to the SAM 2.1 Hiera-small checkpoint (see scripts/fetch_weights.py).",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=Path("docs") / "spike-contact-sheet.png",
        help="Where to save the input|desired|region|SAM contact sheet.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for SAM 2 inference (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--min-area-fraction",
        type=float,
        default=0.0015,
        help="Regions smaller than this fraction of the image area are merged away "
        "(default 0.15%%).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input images, for fast iteration (default: all 7).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to parse instead of ``sys.argv[1:]``, by default None.

    Returns
    -------
    int
        Process exit code: ``0`` on overall PASS, ``1`` on overall FAIL or error.
    """
    args = build_parser().parse_args(argv)

    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    image_paths = _iter_input_images(args.input_dir, limit=args.limit)
    if not image_paths:
        print(f"No starting-image*.jpeg files found in {args.input_dir}", file=sys.stderr)
        return 1

    mask_generator = load_mask_generator(args.checkpoint, device=device)

    results = [
        evaluate_image(path, mask_generator, min_area_fraction=args.min_area_fraction)
        for path in image_paths
    ]
    overall_passed = print_report(results)

    try:
        sheet = build_contact_sheet(
            args.input_dir, args.ref_dir, mask_generator, min_area_fraction=args.min_area_fraction
        )
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.contact_sheet), sheet)
        print(f"Saved contact sheet -> {args.contact_sheet}")
    except ValueError as exc:
        print(f"Skipping contact sheet: {exc}", file=sys.stderr)

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
