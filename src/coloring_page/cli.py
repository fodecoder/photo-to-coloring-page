"""Command-line interface for converting photos into coloring pages.

Built on the standard-library ``argparse`` rather than ``click``/``typer``:
the CLI's surface is small (a handful of flags, no subcommands or shell
completion needs), so pulling in an extra dependency would not earn its
keep here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coloring_page.engines.registry import ENGINES, get_engine
from coloring_page.pipeline import (
    DEFAULT_WORKING_DIMENSION,
    SUPPORTED_INPUT_SUFFIXES,
    UnsupportedFormatError,
    convert_image,
    load_image,
    save_image,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI's argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting the tool's input/output paths and conversion
        options, shared by both single-file and batch (directory) mode.
    """
    parser = argparse.ArgumentParser(
        prog="coloring-page",
        description=(
            "Convert a photo (or a directory of photos) into a printable "
            "black-and-white coloring page."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an input image file, or a directory of images for batch mode.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to write the output image, or an output directory in batch mode.",
    )
    parser.add_argument(
        "--style",
        choices=sorted(ENGINES),
        default="canny",
        help="Conversion style to use (default: canny).",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=1,
        help="Approximate output line thickness in pixels (default: 1).",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=DEFAULT_WORKING_DIMENSION,
        help=(
            "Working resolution: downscale the image so neither side exceeds "
            f"this many pixels before conversion (default: {DEFAULT_WORKING_DIMENSION})."
        ),
    )
    return parser


def _iter_batch_inputs(input_dir: Path) -> list[Path]:
    """Return the supported image files directly inside ``input_dir``, sorted."""
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )


def run(args: argparse.Namespace) -> int:
    """Execute a parsed CLI invocation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments produced by :func:`build_parser`.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on a reported failure.
    """
    engine = get_engine(args.style)

    if args.input.is_dir():
        image_paths = _iter_batch_inputs(args.input)
        if not image_paths:
            print(f"No supported images (.jpg, .jpeg, .png) found in {args.input}", file=sys.stderr)
            return 1

        for image_path in image_paths:
            destination = args.output / f"{image_path.stem}_coloring{image_path.suffix}"
            try:
                image = load_image(image_path)
                result = convert_image(
                    image,
                    engine,
                    line_thickness=args.thickness,
                    max_dimension=args.max_dimension,
                )
                save_image(result, destination)
                print(f"Converted {image_path} -> {destination}")
            except (FileNotFoundError, UnsupportedFormatError, ValueError, OSError) as exc:
                print(f"Skipping {image_path}: {exc}", file=sys.stderr)
        return 0

    try:
        image = load_image(args.input)
    except (FileNotFoundError, UnsupportedFormatError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    result = convert_image(
        image,
        engine,
        line_thickness=args.thickness,
        max_dimension=args.max_dimension,
    )
    save_image(result, args.output)
    print(f"Converted {args.input} -> {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to parse instead of ``sys.argv[1:]``, by default None.
        Mainly useful for testing.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
