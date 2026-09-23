"""Command-line interface for converting photos into coloring pages.

A thin wrapper over :func:`coloring_page.api.convert_image` and
:func:`coloring_page.batch.convert_batch`: this module owns argument
parsing, ``Profile`` assembly, exit codes, and user-facing
success/failure messages, but no conversion logic of its own. Built on
the standard-library ``argparse`` rather than ``click``/``typer``: the
CLI's surface is small (a handful of flags, no subcommands or shell
completion needs), so pulling in an extra dependency would not earn its
keep here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from coloring_page.api import convert_image
from coloring_page.batch import convert_batch
from coloring_page.engines.registry import ENGINES
from coloring_page.exceptions import (
    ColoringPageError,
    EngineUnavailableError,
    ImageTooLargeError,
    QualityGateError,
    UnsupportedImageError,
    WeightsChecksumError,
    WeightsMissingError,
)
from coloring_page.logging_utils import configure_logging
from coloring_page.pipeline import SUPPORTED_INPUT_SUFFIXES
from coloring_page.profile import Profile

logger = logging.getLogger(__name__)

#: Exit code for each typed error, chosen so a caller scripting this CLI
#: can distinguish failure modes without parsing stderr. ``ColoringPageError``
#: itself (a subclass not listed here) falls back to 1.
_EXIT_CODES: dict[type[ColoringPageError], int] = {
    UnsupportedImageError: 2,
    ImageTooLargeError: 3,
    WeightsMissingError: 4,
    WeightsChecksumError: 5,
    EngineUnavailableError: 6,
    QualityGateError: 7,
}


def _exit_code_for(exc: ColoringPageError) -> int:
    """Look up the dedicated exit code for ``exc``, falling back to 1."""
    return _EXIT_CODES.get(type(exc), 1)


def build_parser(argv: list[str] | None = None) -> argparse.ArgumentParser:
    """Construct the CLI's argument parser.

    Parameters
    ----------
    argv : list[str] | None, optional
        The argument list this parser will eventually parse, by default
        None (``sys.argv[1:]``). Only used here to pre-parse
        ``--show-experimental`` so ``--style``'s displayed choices can
        depend on it -- see that argument's ``metavar`` below.

    Returns
    -------
    argparse.ArgumentParser
        Parser accepting the tool's input/output paths and conversion
        options, shared by both single-file and batch (directory) mode.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--show-experimental", action="store_true")
    show_experimental = pre_parser.parse_known_args(argv)[0].show_experimental

    # Experimental (superseded baseline) styles stay fully usable via
    # --style <name> regardless of this flag -- `choices` below is never
    # narrowed, only the *displayed* list is, via `metavar`. Narrowing
    # `choices` itself would make an experimental style like `canny`
    # rejected as an "invalid choice" unless --show-experimental were
    # also passed, which would be a real usability regression, not just
    # a cosmetic one.
    visible_styles = sorted(
        name for name, cls in ENGINES.items() if show_experimental or not cls.experimental
    )

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
        "--profile",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file produced by Profile.to_json(), used as the base "
            "configuration. Any other flag below that's explicitly passed overrides "
            "the corresponding field."
        ),
    )
    parser.add_argument(
        "--style",
        choices=sorted(ENGINES),
        metavar="{" + ",".join(visible_styles) + "}",
        default=None,
        help=(
            "Conversion style to use (default: lineart-raster). Measured "
            "with scripts/ablation.py against this project's reference "
            "photos: lineart-raster stays close to the target ink-coverage "
            "band at every tested resolution, where the previous default, "
            "chained (now experimental, hidden below -- see "
            "--show-experimental), overshot it 2-3x with hundreds of "
            "noise-sized 'enclosed regions' per image. lineart-raster "
            "requires the lineart_raster extra and a manually downloaded "
            "checkpoint (see the README); without them this raises "
            "EngineUnavailableError/WeightsMissingError -- pass a "
            "zero-dependency style (e.g. --style canny) if you haven't "
            "installed it."
        ),
    )
    parser.add_argument(
        "--show-experimental",
        action="store_true",
        help="Also list experimental (superseded baseline) conversion styles above.",
    )
    parser.add_argument(
        "--detail",
        choices=["toddler", "child", "adult"],
        default=None,
        help=(
            "How much fine structure survives into the output (default: child). "
            "The real product parameter -- see the README's profile table."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Inference device for ML-backed styles; ignored by classical styles (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible output (default: 0). Pass a value explicitly to change it.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Reject inputs larger than this many pixels (default: 40,000,000).",
    )
    parser.add_argument(
        "--format",
        choices=["svg", "pdf", "png"],
        default=None,
        help="Output format (default: inferred from the output path's extension, else png).",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=None,
        help=(
            "Deprecated, ignored: pixel-based line thickness has been replaced by "
            "--detail, which expresses the same idea in physical (mm) terms "
            "independent of working resolution. Kept only so existing invocations "
            "still parse; passing it logs a warning and has no effect."
        ),
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=None,
        help="Deprecated, ignored: working resolution is now fixed internally.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Validate output colorability (ink coverage, enclosed/leaking "
            "regions, dangling contour endpoints -- see coloring_page.validate) "
            "and exit with status 7 if it fails quality thresholds."
        ),
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help=(
            "Write intermediate conversion stages (for engines that report "
            "them, plus the pre/post detail-filter drawing) into this directory. "
            "In batch mode, each input file gets its own subdirectory by stem."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=(
            "Worker processes for batch mode (default: min(4, cpu_count), or 1 for "
            "styles that hold a model resident in VRAM, which always run serially "
            "regardless of this flag)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit debug-level logging and full tracebacks on error.",
    )
    return parser


def _iter_batch_inputs(input_dir: Path) -> list[Path]:
    """Return the supported image files directly inside ``input_dir``, sorted."""
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )


def _build_profile(args: argparse.Namespace) -> Profile:
    """Assemble a ``Profile`` from ``--profile`` (if given) and individual flags.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    Profile
        The base profile from ``--profile``, if given, else
        ``Profile()``, with any explicitly-passed flag overriding the
        corresponding field.
    """
    base = Profile.from_json(args.profile.read_text()) if args.profile is not None else Profile()

    overrides: dict[str, object] = {}
    if args.style is not None:
        overrides["style"] = args.style
    if args.detail is not None:
        overrides["detail"] = args.detail
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.max_pixels is not None:
        overrides["max_pixels"] = args.max_pixels
    if args.debug_dir is not None:
        overrides["debug_dir"] = args.debug_dir

    return base.with_overrides(**overrides) if overrides else base


def _resolve_format(args: argparse.Namespace) -> str:
    """Pick an output format: ``--format`` if given, else inferred from ``output``'s suffix."""
    if args.format is not None:
        return str(args.format)
    suffix = args.output.suffix.lower().lstrip(".")
    return suffix if suffix in ("svg", "pdf", "png") else "png"


def run(args: argparse.Namespace) -> int:
    """Execute a parsed CLI invocation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments produced by :func:`build_parser`.

    Returns
    -------
    int
        Process exit code: ``0`` on success, or one of :data:`_EXIT_CODES`
        (``1`` for an untyped ``ColoringPageError``) on failure.
    """
    if args.thickness is not None:
        logger.warning(
            "--thickness is deprecated and has no effect; use --detail instead "
            "(default: child). See the README's profile table."
        )
    if args.max_dimension is not None:
        logger.warning("--max-dimension is deprecated and has no effect.")

    try:
        profile = _build_profile(args)
    except ColoringPageError as exc:
        logger.error(str(exc))
        return _exit_code_for(exc)

    fmt = _resolve_format(args)

    if args.input.is_dir():
        image_paths = _iter_batch_inputs(args.input)
        if not image_paths:
            print(f"No supported images (.jpg, .jpeg, .png) found in {args.input}", file=sys.stderr)
            return 1

        try:
            summary = convert_batch(
                image_paths,
                args.output,
                profile=profile,
                fmt=fmt,  # type: ignore[arg-type]
                strict=args.strict,
                jobs=args.jobs,
            )
        except ColoringPageError as exc:
            logger.error(str(exc))
            return _exit_code_for(exc)
        for item in summary.failed:
            print(f"Skipping {item.input_path}: {item.error}", file=sys.stderr)
        for item in summary.succeeded:
            print(f"Converted {item.input_path} -> {item.output_path}")

        any_strict_failure = any(r.passed_strict is False for r in summary.results)
        if summary.failed:
            return 1
        return _EXIT_CODES[QualityGateError] if any_strict_failure else 0

    try:
        result = convert_image(args.input, profile=profile)
    except (ColoringPageError, FileNotFoundError, OSError) as exc:
        logger.error(str(exc))
        if args.verbose:
            logger.exception("Full traceback:")
        return _exit_code_for(exc) if isinstance(exc, ColoringPageError) else 1

    if args.strict and not result.report.passed:
        logger.error(f"Failed --strict validation: {result.report}")
        return _EXIT_CODES[QualityGateError]

    getattr(result, f"save_{fmt}")(args.output)
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
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
