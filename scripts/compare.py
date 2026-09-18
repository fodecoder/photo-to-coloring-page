"""Run every registered engine on a folder of images and compare results.

Writes, per input image, a labeled contact sheet showing every style's
output side by side, plus a single ``metrics.csv`` covering every
(image, style) pair. This is the empirical decision mechanism for
picking a default conversion style: run this against a handful of real
photos, inspect the contact sheets and the metrics table, and only then
update ``cli.py``'s ``--style`` default -- not something to decide by
reading engine source.

Usage
-----
::

    python scripts/compare.py <input_dir> <output_dir> [--styles canny,xdog]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from coloring_page.engines.registry import ENGINES, get_engine
from coloring_page.metrics import (
    BoundaryMetrics,
    compare_boundaries,
    compute_metrics,
    degenerate_floor,
)
from coloring_page.pipeline import SUPPORTED_INPUT_SUFFIXES, convert_image, load_image

#: Height, in pixels, of the label strip drawn above each contact-sheet tile.
_LABEL_HEIGHT = 24

#: Input image filename -> matching reference drawing filename, both resolved
#: relative to ``--ref-dir``. These are the only pairs with a known
#: correspondence; not every input image has one.
REFERENCE_PAIRS = {
    "starting-image.jpeg": "desired.jpg",
    "starting-image-5.jpeg": "desired-5.jpg",
    "starting-image-7.jpeg": "desired-7.jpg",
}

#: Styles compared by default. Excludes ``adaptive``: ink coverage runs
#: several times this project's 3-7% admissibility band (a texturized
#: sketch effect, not a coloring-page style -- see the README). Excludes
#: ``gated``: measured boundary F1 does not currently beat plain
#: ``chained`` at any non-trivial gate threshold (see ``engines/gated.py``'s
#: docstring). Excludes ``xdog``: measured ``f1_normalized`` averages
#: ~0.05 across this project's 3 reference pairs (recall 0.32-0.42 even
#: though the DoG sign-error bug documented in
#: ``docs/IMPROVEMENT-PROMPT.md`` is already fixed -- it now just draws
#: too little ink relative to the reference to be a coloring-page
#: candidate here, not a leftover bug). All three stay registered and
#: selectable via ``--styles`` for experimentation, just not part of the
#: default comparison.
RECOMMENDED_STYLES = sorted(ENGINES.keys() - {"adaptive", "gated", "xdog"})


def _iter_input_images(input_dir: Path) -> list[Path]:
    """Return the supported image files directly inside ``input_dir``, sorted."""
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )


def _label_tile(image: np.ndarray, label: str) -> np.ndarray:
    """Stack a text label above a grayscale tile, returning a BGR image."""
    tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    strip = np.full((_LABEL_HEIGHT, tile.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        strip,
        label,
        (4, _LABEL_HEIGHT - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([strip, tile])


def _convert_with_style(image: np.ndarray, style: str) -> np.ndarray | None:
    """Run one style, returning ``None`` (and a stderr warning) on failure.

    A style can fail per-image for reasons unrelated to the comparison
    itself -- most notably ``anime2sketch`` raising ``FileNotFoundError``
    when its pretrained weights haven't been downloaded. One missing
    style shouldn't abort comparing the rest.
    """
    try:
        engine = get_engine(style)
        return convert_image(image, engine)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Skipping style {style!r}: {exc}", file=sys.stderr)
        return None


def _aspect_ratio(shape: tuple[int, ...]) -> float:
    """Long-side-over-short-side ratio for a ``(H, W, ...)`` shape."""
    height, width = shape[0], shape[1]
    return max(height, width) / min(height, width)


def _reference_boundary_scores(
    image_name: str, result: np.ndarray, ref_dir: Path, aspect_tolerance: float
) -> BoundaryMetrics | None:
    """Score ``result`` against its reference drawing, if one is known and available.

    Returns ``None`` (with a stderr note) when ``image_name`` has no known
    pair or the reference file is missing -- this is a data-availability
    gap, not a mismatch, so it's silent apart from the note. When a pair
    *is* found but the two images' aspect ratios differ by more than
    ``aspect_tolerance``, this still scores it (they're close crops, not a
    wrong-image mixup) but prints a warning so the resulting F1 is read as
    approximate rather than exact.
    """
    ref_name = REFERENCE_PAIRS.get(image_name)
    if ref_name is None:
        return None

    ref_path = ref_dir / ref_name
    gt = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        print(
            f"No reference drawing at {ref_path}; skipping boundary F1 for {image_name}",
            file=sys.stderr,
        )
        return None

    pred_ratio = _aspect_ratio(result.shape)
    gt_ratio = _aspect_ratio(gt.shape)
    if abs(pred_ratio - gt_ratio) > aspect_tolerance:
        print(
            f"Aspect ratio mismatch for {image_name} vs {ref_name} "
            f"({pred_ratio:.3f} vs {gt_ratio:.3f}); boundary F1 is approximate.",
            file=sys.stderr,
        )

    return compare_boundaries(result, gt)


def build_contact_sheet(image: np.ndarray, styles: list[str]) -> np.ndarray:
    """Convert ``image`` with every style in ``styles`` and lay results out side by side.

    Parameters
    ----------
    image : np.ndarray
        BGR input image, as returned by :func:`coloring_page.pipeline.load_image`.
    styles : list[str]
        Style names (keys of ``ENGINES``) to run and compare.

    Returns
    -------
    np.ndarray
        A single BGR contact-sheet image, one labeled tile per style that
        succeeded (styles that raised an error are omitted).
    """
    tiles = []
    for style in styles:
        result = _convert_with_style(image, style)
        if result is not None:
            tiles.append(_label_tile(result, style))

    if not tiles:
        raise ValueError("Every style failed to convert this image; nothing to compare.")

    max_height = max(tile.shape[0] for tile in tiles)
    padded_tiles = [
        cv2.copyMakeBorder(
            tile, 0, max_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        for tile in tiles
    ]
    return cv2.hconcat(padded_tiles)


def compare(
    input_dir: Path,
    output_dir: Path,
    styles: list[str],
    *,
    ref_dir: Path | None = None,
    aspect_tolerance: float = 0.05,
) -> None:
    """Run ``styles`` over every image in ``input_dir`` and write comparison output.

    Parameters
    ----------
    input_dir : Path
        Directory of ``.jpg``/``.jpeg``/``.png`` images to compare styles on.
    output_dir : Path
        Directory to write contact-sheet PNGs and ``metrics.csv`` into.
    styles : list[str]
        Style names (keys of ``ENGINES``) to run and compare.
    ref_dir : Path | None, optional
        Directory to look up reference drawings in (see
        ``REFERENCE_PAIRS``), by default None, which disables boundary F1
        scoring entirely (images without a known reference always leave
        those columns blank regardless).
    aspect_tolerance : float, optional
        Maximum long/short aspect-ratio difference, between a result and
        its reference, allowed before printing a mismatch warning, by
        default 0.05.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _iter_input_images(input_dir)

    if ref_dir is not None:
        for ref_name in REFERENCE_PAIRS.values():
            gt = cv2.imread(str(ref_dir / ref_name), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            long_side = max(gt.shape)
            tolerance = max(2, round(0.005 * long_side))
            floor = degenerate_floor(gt, tolerance=tolerance)
            print(
                f"Degenerate floor for {ref_name} (tolerance={tolerance}px): {floor:.4f} "
                "-- any f1_normalized is relative to this, not to zero.",
                file=sys.stderr,
            )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["image", "style", "ink_coverage", "component_count", "median_stroke_length",
             "noise_fraction", "ref_precision", "ref_recall", "ref_f1", "ref_f1_normalized",
             "ink_admissible"]
        )  # fmt: skip

        for image_path in image_paths:
            image = load_image(image_path)
            sheet = build_contact_sheet(image, styles)
            cv2.imwrite(str(output_dir / f"{image_path.stem}_compare.png"), sheet)

            for style in styles:
                result = _convert_with_style(image, style)
                if result is None:
                    continue
                metrics = compute_metrics(result)

                ref_scores = (
                    _reference_boundary_scores(image_path.name, result, ref_dir, aspect_tolerance)
                    if ref_dir is not None
                    else None
                )
                ref_row = (
                    [
                        f"{ref_scores.precision:.4f}",
                        f"{ref_scores.recall:.4f}",
                        f"{ref_scores.f1:.4f}",
                        f"{ref_scores.f1_normalized:.4f}",
                        str(ref_scores.ink_admissible),
                    ]
                    if ref_scores is not None
                    else ["", "", "", "", ""]
                )

                writer.writerow(
                    [
                        image_path.name,
                        style,
                        f"{metrics.ink_coverage:.4f}",
                        metrics.component_count,
                        f"{metrics.median_stroke_length:.2f}",
                        f"{metrics.noise_fraction:.4f}",
                        *ref_row,
                    ]
                )


def build_parser() -> argparse.ArgumentParser:
    """Construct this script's argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare conversion styles side by side on a folder of images."
    )
    parser.add_argument("input_dir", type=Path, help="Directory of input images.")
    parser.add_argument("output_dir", type=Path, help="Directory to write comparison output into.")
    parser.add_argument(
        "--styles",
        type=str,
        default=None,
        help="Comma-separated style names to compare (default: RECOMMENDED_STYLES, i.e. all "
        "registered styles except 'adaptive', 'gated', and 'xdog').",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path("docs"),
        help="Directory to look up reference drawings (REFERENCE_PAIRS) in, by default 'docs'.",
    )
    parser.add_argument(
        "--aspect-tolerance",
        type=float,
        default=0.05,
        help="Max long/short aspect-ratio difference before warning of a reference mismatch.",
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
        Process exit code: ``0`` on success, ``1`` if no input images were found.
    """
    args = build_parser().parse_args(argv)
    styles = RECOMMENDED_STYLES if args.styles is None else args.styles.split(",")

    image_paths = _iter_input_images(args.input_dir)
    if not image_paths:
        print(f"No supported images found in {args.input_dir}", file=sys.stderr)
        return 1

    compare(
        args.input_dir,
        args.output_dir,
        styles,
        ref_dir=args.ref_dir,
        aspect_tolerance=args.aspect_tolerance,
    )
    print(f"Compared styles {styles} on {len(image_paths)} image(s) -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
