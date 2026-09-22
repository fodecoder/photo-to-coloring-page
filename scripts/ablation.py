"""Measure the standalone `lineart-raster` engine against the `chained` baseline.

Phase 2 of the raster-artwork work: `lineart_raster.LineArtRasterEngine`
isolates `lineart.py`'s Stage B (the controlnet_aux detail-line network)
as its own raster-only engine, kept deliberately un-vectorized -- see
`coloring_page.artwork` for why. This script is the measurement gate for
that engine: run it at a handful of resolutions against `chained` (today's
CLI default) on every `docs/starting-image*.jpeg`, print a metrics table,
and produce one contact sheet plus one fixed-crop detail sheet so the
comparison isn't judged only at thumbnail scale.

This script does not decide anything by itself -- its output is meant to
be inspected (numbers + images) before any decision to change the CLI
default or recombine this engine with SAM 2 segmentation (out of scope
here).

Usage
-----
::

    python scripts/ablation.py [--input-dir docs] [--output docs/ablation-contact-sheet.png]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from coloring_page.artwork import RasterArtwork
from coloring_page.drawing import Drawing, rasterize
from coloring_page.engines.registry import get_engine
from coloring_page.page import PageSpec
from coloring_page.pipeline import SUPPORTED_INPUT_SUFFIXES, load_image
from coloring_page.validate import validate

#: (style, resolution) pairs to run per image. `resolution=None` for
#: `chained` since it isn't a `resolution`-parameterized engine --
#: `get_engine` simply ignores the argument when unsupported.
CONFIGS: list[tuple[str, int | None]] = [
    ("lineart-raster", 512),
    ("lineart-raster", 768),
    ("lineart-raster", 1024),
    ("lineart-raster", 1280),
    ("chained", None),
]

#: Height, in pixels, of the label strip drawn above each contact-sheet tile.
_LABEL_HEIGHT = 24

#: Long side, in pixels, tiles are resized to for the contact sheet --
#: independent of each config's own working resolution, so every column
#: is directly comparable at a shared display size.
_PREVIEW_LONG_SIDE = 480

#: Fixed detail-crop box per input filename, as `(x0, y0, x1, y1)`
#: fractions of the *source* image's own width/height. Centered,
#: moderate-zoom placeholders -- pick a region that actually contains a
#: face, hands, or another fine-detail subject once the real
#: `docs/starting-image*.jpeg` content has been inspected visually; a
#: crop box that lands on flat background defeats the point of this sheet
#: (checking detail fidelity, not overall composition).
DETAIL_CROPS: dict[str, tuple[float, float, float, float]] = {}
_DEFAULT_DETAIL_CROP = (0.35, 0.35, 0.65, 0.65)


@dataclass(frozen=True)
class AblationRow:
    """One (image, config) measurement."""

    image_name: str
    style: str
    resolution: int | None
    elapsed_s: float
    ink_coverage: float
    antialiased_fraction: float | None
    enclosed_regions: int
    region_area_mm2_p5: float | None


def _iter_input_images(input_dir: Path) -> list[Path]:
    """Return the supported image files directly inside ``input_dir``, sorted."""
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )


def _preview_image(artwork: Drawing | RasterArtwork, *, long_side_px: int) -> np.ndarray:
    """Render an ``Artwork`` to a plain grayscale preview at a fixed display size.

    Deliberately not ``render.to_png``'s full-page rendering (with margins
    and a fixed physical DPI) -- this is for a contact sheet, where every
    tile should fill its frame edge to edge and be directly comparable at
    one shared pixel size, not a print-page mockup.
    """
    if isinstance(artwork, RasterArtwork):
        height, width = artwork.image.shape[:2]
        if height >= width:
            target_h, target_w = long_side_px, max(1, round(long_side_px * width / height))
        else:
            target_w, target_h = long_side_px, max(1, round(long_side_px * height / width))
        return cv2.resize(artwork.image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return rasterize(artwork, long_side_px=long_side_px)


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


def _row_of_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    """Horizontally concatenate tiles of possibly-differing heights, white-padded."""
    max_height = max(tile.shape[0] for tile in tiles)
    padded = [
        cv2.copyMakeBorder(
            tile, 0, max_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        for tile in tiles
    ]
    return cv2.hconcat(padded)


def _config_label(style: str, resolution: int | None) -> str:
    return style if resolution is None else f"{style}@{resolution}"


def _detail_crop_box(image_name: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Resolve a detail-crop box, in source-image pixels, for ``image_name``."""
    x0f, y0f, x1f, y1f = DETAIL_CROPS.get(image_name, _DEFAULT_DETAIL_CROP)
    height, width = shape
    return (
        round(x0f * width),
        round(y0f * height),
        round(x1f * width),
        round(y1f * height),
    )


def run_ablation(input_dir: Path) -> tuple[list[AblationRow], dict[Path, list[np.ndarray]]]:
    """Run every ``CONFIGS`` entry over every image in ``input_dir``.

    Returns
    -------
    tuple[list[AblationRow], dict[Path, list[np.ndarray]]]
        The measurement rows, and each image's list of raw (unlabeled)
        preview tiles in ``CONFIGS`` order -- reused to build both the
        overview contact sheet and the detail-crop sheet without
        re-running any engine.
    """
    rows: list[AblationRow] = []
    previews: dict[Path, list[np.ndarray]] = {}

    for image_path in _iter_input_images(input_dir):
        image = load_image(image_path)
        tiles: list[np.ndarray] = []

        for style, resolution in CONFIGS:
            engine = get_engine(style, resolution=resolution)
            start = time.perf_counter()
            artwork = engine.convert(image)
            elapsed = time.perf_counter() - start

            report = validate(artwork, PageSpec())
            rows.append(
                AblationRow(
                    image_name=image_path.name,
                    style=style,
                    resolution=resolution,
                    elapsed_s=elapsed,
                    ink_coverage=report.ink_coverage,
                    antialiased_fraction=report.antialiased_fraction,
                    enclosed_regions=report.enclosed_regions,
                    region_area_mm2_p5=report.region_area_mm2_p5,
                )
            )
            tiles.append(_preview_image(artwork, long_side_px=_PREVIEW_LONG_SIDE))

        previews[image_path] = tiles

    return rows, previews


def print_table(rows: list[AblationRow]) -> None:
    """Print a plain-text results table to stdout."""
    header = (
        f"{'image':<24}{'config':<22}{'time_s':>8}{'ink_cov':>10}"
        f"{'aa_frac':>10}{'enclosed':>10}{'p5_area_mm2':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        aa = "n/a" if row.antialiased_fraction is None else f"{row.antialiased_fraction:.4f}"
        p5 = "n/a" if row.region_area_mm2_p5 is None else f"{row.region_area_mm2_p5:.2f}"
        print(
            f"{row.image_name:<24}{_config_label(row.style, row.resolution):<22}"
            f"{row.elapsed_s:>8.2f}{row.ink_coverage:>10.4f}{aa:>10}"
            f"{row.enclosed_regions:>10}{p5:>13}"
        )


def build_contact_sheet(previews: dict[Path, list[np.ndarray]]) -> np.ndarray:
    """Lay out every image's config previews as rows x columns, labeled."""
    labels = [_config_label(style, resolution) for style, resolution in CONFIGS]
    sheet_rows = []
    for tiles in previews.values():
        labeled = [_label_tile(tile, label) for tile, label in zip(tiles, labels, strict=True)]
        row_tile = _row_of_tiles(labeled)
        row_tile = cv2.copyMakeBorder(
            row_tile, 0, 4, 0, 0, cv2.BORDER_CONSTANT, value=(200, 200, 200)
        )
        sheet_rows.append(row_tile)

    max_width = max(row.shape[1] for row in sheet_rows)
    padded_rows = [
        cv2.copyMakeBorder(
            row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        for row in sheet_rows
    ]
    return cv2.vconcat(padded_rows)


def build_crop_sheet(input_dir: Path, previews: dict[Path, list[np.ndarray]]) -> np.ndarray:
    """Lay out a 100% detail crop of each config's *source-resolution* output.

    Crops the full-resolution artwork (not the shared-size preview tiles
    :func:`build_contact_sheet` uses), since a crop of an already-downscaled
    preview would hide exactly the resolution differences this ablation
    exists to compare.
    """
    labels = [_config_label(style, resolution) for style, resolution in CONFIGS]
    sheet_rows = []
    for image_path in previews:
        image = load_image(image_path)
        x0, y0, x1, y1 = _detail_crop_box(image_path.name, image.shape[:2])

        tiles = []
        for style, resolution in CONFIGS:
            engine = get_engine(style, resolution=resolution)
            artwork = engine.convert(image)
            full = (
                artwork.image
                if isinstance(artwork, RasterArtwork)
                else rasterize(artwork, long_side_px=max(image.shape[:2]))
            )
            # `rasterize`'s output resolution is `long_side_px`, not
            # necessarily image.shape -- rescale the crop box proportionally.
            fh, fw = full.shape[:2]
            sh, sw = image.shape[:2]
            crop = full[
                round(y0 * fh / sh) : round(y1 * fh / sh),
                round(x0 * fw / sw) : round(x1 * fw / sw),
            ]
            tiles.append(crop)

        labeled = [_label_tile(tile, label) for tile, label in zip(tiles, labels, strict=True)]
        sheet_rows.append(_row_of_tiles(labeled))

    max_width = max(row.shape[1] for row in sheet_rows)
    padded_rows = [
        cv2.copyMakeBorder(
            row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        for row in sheet_rows
    ]
    return cv2.vconcat(padded_rows)


def build_parser() -> argparse.ArgumentParser:
    """Construct this script's argument parser."""
    parser = argparse.ArgumentParser(
        description="Measure lineart-raster at several resolutions against the chained baseline."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("docs"), help="Directory of input images."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/ablation-contact-sheet.png"),
        help="Path to write the overview contact sheet to.",
    )
    parser.add_argument(
        "--crop-output",
        type=Path,
        default=Path("docs/ablation-crops.png"),
        help="Path to write the 100%% detail-crop sheet to.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` if no input images were found.
    """
    args = build_parser().parse_args(argv)

    image_paths = _iter_input_images(args.input_dir)
    if not image_paths:
        print(f"No supported images found in {args.input_dir}", file=sys.stderr)
        return 1

    rows, previews = run_ablation(args.input_dir)
    print_table(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), build_contact_sheet(previews))
    cv2.imwrite(str(args.crop_output), build_crop_sheet(args.input_dir, previews))
    print(f"\nWrote {args.output} and {args.crop_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
