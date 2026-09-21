"""Measure peak RSS for a conversion, to feed the README's hardware-requirements table.

Not part of ``src/`` and not covered by the test suite (a resource-usage
measurement, not a behavior to assert on in CI) -- a dev-only tool run by
hand to produce numbers for documentation. Uses ``resource.getrusage`` on
POSIX (accurate, cheap) and falls back to ``tracemalloc`` (Python-object
allocations only -- excludes OpenCV/torch native buffers, so it
undercounts, but it's the best portable option) on Windows.

Usage
-----
::

    python scripts/profile_memory.py path/to/large_photo.jpg --style chained
    python scripts/profile_memory.py path/to/large_photo.jpg --style region --style lineart
"""

from __future__ import annotations

import argparse
import platform
import sys
import tracemalloc
from pathlib import Path

from coloring_page.api import convert_image
from coloring_page.profile import Profile


def _peak_rss_mb_posix() -> float | None:
    """Peak resident set size, in MB, since process start (POSIX only)."""
    try:
        import resource
    except ImportError:
        return None
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS -- normalize via platform.
    return peak_kb / 1024 if platform.system() == "Linux" else peak_kb / (1024 * 1024)


def profile_conversion(image_path: Path, style: str) -> None:
    """Run one conversion through ``style`` and print its peak memory usage."""
    profile = Profile(style=style)

    if platform.system() != "Windows":
        result = convert_image(image_path, profile=profile)
        peak_mb = _peak_rss_mb_posix()
        print(f"{style}: peak RSS = {peak_mb:.1f} MB" if peak_mb is not None else f"{style}: n/a")
        del result
        return

    tracemalloc.start()
    result = convert_image(image_path, profile=profile)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"{style}: peak Python-object allocation = {peak_bytes / (1024 * 1024):.1f} MB "
        "(tracemalloc; excludes native OpenCV/torch buffers, so this undercounts)"
    )
    del result


def build_parser() -> argparse.ArgumentParser:
    """Construct this script's argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the input photo to profile.")
    parser.add_argument(
        "--style", action="append", default=[], help="Style to profile (repeatable)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point."""
    args = build_parser().parse_args(argv)
    styles = args.style or ["chained"]
    for style in styles:
        profile_conversion(args.image, style)
    return 0


if __name__ == "__main__":
    sys.exit(main())
