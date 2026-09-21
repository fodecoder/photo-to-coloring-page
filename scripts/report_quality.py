"""Run ``validate()`` over the project's reference images and print a markdown table.

Replaces the old boundary-F1 comparison (``scripts/metrics.py``,
``scripts/compare.py``) as the number this project's README reports,
matching this project's actual acceptance gate (``--strict``,
:func:`coloring_page.validate.validate`) instead of similarity to an
artistic reference -- see ``docs/DIAGNOSIS.md`` §4 for why the latter is a
poor *acceptance* criterion.

Usage
-----
::

    python scripts/report_quality.py docs --style chained
    python scripts/report_quality.py docs --style chained --style region
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coloring_page.api import convert_image
from coloring_page.pipeline import SUPPORTED_INPUT_SUFFIXES
from coloring_page.profile import Profile

#: Reference input photos this project measures against, relative to
#: --ref-dir. Matches scripts/compare.py's REFERENCE_PAIRS keys plus the
#: unpaired starting-image-N files that have no hand-drawn counterpart.
REFERENCE_IMAGES = [
    "starting-image.jpeg",
    "starting-image-2.jpeg",
    "starting-image-3.jpeg",
    "starting-image-4.jpeg",
    "starting-image-5.jpeg",
    "starting-image-6.jpeg",
    "starting-image-7.jpeg",
]


def build_parser() -> argparse.ArgumentParser:
    """Construct this script's argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_dir", type=Path, help="Directory containing the reference images.")
    parser.add_argument(
        "--style",
        action="append",
        default=[],
        help="Style to measure (repeatable; default: chained).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point."""
    args = build_parser().parse_args(argv)
    styles = args.style or ["chained"]

    print(
        "| style | image | ink_coverage | enclosed | leaking | min_region_mm2 | dangling | passed |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for style in styles:
        for filename in REFERENCE_IMAGES:
            path = args.ref_dir / filename
            if not path.exists() or path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
                continue
            result = convert_image(path, profile=Profile(style=style))
            report = result.report
            min_area = f"{report.min_region_area_mm2:.1f}" if report.min_region_area_mm2 else "n/a"
            print(
                f"| {style} | {filename} | {report.ink_coverage:.3f} | "
                f"{report.enclosed_regions} | {report.leaking_regions} | {min_area} | "
                f"{report.dangling_endpoints} | {'yes' if report.passed else 'no'} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
