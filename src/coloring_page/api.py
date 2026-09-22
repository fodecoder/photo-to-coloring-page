"""Public library entrypoint: ``convert_image(src, *, profile) -> Result``.

Everything that used to live only in ``cli.py`` -- resizing, engine
dispatch, quality validation -- is orchestrated here instead, so a caller
embedding this package in a service gets the same behavior the CLI does
without going through argparse or the filesystem. No conversion engine
performs file I/O; the one place this function touches disk is
:func:`coloring_page.pipeline.load_image`, called here, once, when ``src``
is a ``Path``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from coloring_page.drawing import Drawing, rasterize
from coloring_page.engines.registry import get_engine
from coloring_page.exceptions import ImageTooLargeError, UnsupportedImageError
from coloring_page.logging_utils import stage_timer
from coloring_page.pipeline import (
    DEFAULT_WORKING_DIMENSION,
    DebugSink,
    load_image,
    resize_to_max_dimension,
)
from coloring_page.profile import RASTER_RESOLUTION_PRESETS, Profile, apply_detail
from coloring_page.result import Result
from coloring_page.seeding import seed_everything
from coloring_page.validate import validate

logger = logging.getLogger(__name__)


def _validate_ndarray(image: np.ndarray) -> None:
    """Check that an in-memory image matches the BGR ``uint8`` contract every engine expects.

    Parameters
    ----------
    image : np.ndarray

    Raises
    ------
    UnsupportedImageError
        If ``image`` is not a 3-dimensional ``(H, W, 3)`` ``uint8`` array.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise UnsupportedImageError(
            "In-memory images must be a BGR uint8 array with shape (H, W, 3); "
            f"got shape={image.shape}, dtype={image.dtype}."
        )


def _check_max_pixels(image: np.ndarray, max_pixels: int) -> None:
    """Reject an image larger than ``max_pixels`` before any conversion work happens.

    Parameters
    ----------
    image : np.ndarray
    max_pixels : int

    Raises
    ------
    ImageTooLargeError
    """
    height, width = image.shape[:2]
    pixel_count = height * width
    if pixel_count > max_pixels:
        raise ImageTooLargeError(
            f"Input is {pixel_count:,} pixels ({width}x{height}), exceeding the "
            f"configured limit of {max_pixels:,}."
        )


def convert_image(src: Path | np.ndarray, *, profile: Profile) -> Result:
    """Convert a photo into a coloring page, following ``profile``.

    Parameters
    ----------
    src : Path | np.ndarray
        Either a path to a JPG/PNG file on disk, or an already-loaded BGR
        ``uint8`` image array with shape ``(H, W, 3)``. No file I/O
        happens when an array is passed.
    profile : Profile
        Conversion configuration -- style, detail level, page geometry,
        device, seed, and resource limits. See :class:`~coloring_page.profile.Profile`.

    Returns
    -------
    Result
        Exposes ``drawing`` (a ``Drawing`` or a ``RasterArtwork``, see
        :mod:`coloring_page.artwork`), ``report``, ``timings``, and
        ``save_svg``/``save_pdf``/``save_png``. Always returned regardless
        of whether ``report.passed`` -- callers decide for themselves
        whether a failing quality report is acceptable; only the CLI's
        ``--strict`` turns that into a hard error.

    Raises
    ------
    UnsupportedImageError
        If ``src`` is a path OpenCV can't decode, has an unsupported
        extension, or is an in-memory array not matching the BGR
        ``uint8`` ``(H, W, 3)`` contract.
    ImageTooLargeError
        If the image exceeds ``profile.max_pixels``.
    EngineUnavailableError
        If ``profile.style`` isn't a registered engine.
    WeightsMissingError, WeightsChecksumError
        If ``profile.style`` is an ML-backed engine whose checkpoint is
        missing or fails verification.
    """
    timings: dict[str, float] = {}

    with stage_timer(logger, "seed", timings):
        seed_everything(profile.seed)

    with stage_timer(logger, "load", timings):
        if isinstance(src, Path):
            image = load_image(src)
        else:
            _validate_ndarray(src)
            image = src

    with stage_timer(logger, "size_check", timings):
        _check_max_pixels(image, profile.max_pixels)

    with stage_timer(logger, "resize", timings):
        resized = resize_to_max_dimension(image, DEFAULT_WORKING_DIMENSION)
        working_dimension = max(resized.shape[:2])

    debug_sink = DebugSink(profile.debug_dir) if profile.debug_dir is not None else None

    with stage_timer(logger, "engine", timings):
        engine = get_engine(
            profile.style,
            device=profile.device,
            resolution=RASTER_RESOLUTION_PRESETS[profile.detail],
        )
        artwork = engine.convert(resized, debug=debug_sink)

    with stage_timer(logger, "detail_filter", timings):
        # apply_detail is a Drawing-only post-hoc filter (region-area/
        # stroke-length/simplify-epsilon pruning on traced paths); a
        # RasterArtwork has no paths to filter, so it passes through
        # unchanged. See coloring_page.artwork for why both representations
        # exist.
        if isinstance(artwork, Drawing):
            if debug_sink is not None:
                pre_raster = rasterize(artwork, long_side_px=working_dimension)
                debug_sink.save("drawing_pre_filter", pre_raster)
            artwork = apply_detail(artwork, profile, debug=debug_sink)
            if debug_sink is not None:
                post_raster = rasterize(artwork, long_side_px=working_dimension)
                debug_sink.save("drawing_post_filter", post_raster)

    with stage_timer(logger, "validate", timings):
        report = validate(artwork, profile.page)

    timings["total"] = sum(timings.values())
    return Result(drawing=artwork, report=report, timings=timings, page=profile.page)
