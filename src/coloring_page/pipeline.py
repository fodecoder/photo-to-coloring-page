"""Shared image I/O and pre/post-processing used by all conversion engines."""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine

#: File extensions accepted as input photos.
SUPPORTED_INPUT_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

#: Longest-side pixel size that conversion is normalized to before any
#: engine runs. Every engine's resolution-dependent kernel sizes are
#: derived from this value (see :func:`derive_kernel_size`) rather than
#: hardcoded, so a photo shot at any resolution is processed at a scale
#: those kernels were actually tuned for. Chosen in the 1200-1600 range:
#: large enough to preserve the fine linework of an illustrated page,
#: small enough that paper grain/JPEG noise doesn't dwarf fixed-fraction
#: kernels the way it does at full photo resolution (e.g. 2048px).
DEFAULT_WORKING_DIMENSION = 1400


class UnsupportedFormatError(ValueError):
    """Raised when an input file's extension is not a supported image format."""


def load_image(path: Path) -> np.ndarray:
    """Load a photo from disk as a BGR ``numpy`` array.

    Parameters
    ----------
    path : Path
        Path to a JPG or PNG image on disk.

    Returns
    -------
    np.ndarray
        Image array with shape ``(H, W, 3)`` and dtype ``uint8``, in BGR
        channel order (matching ``cv2.imread``'s convention).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    UnsupportedFormatError
        If ``path``'s extension is not one of ``SUPPORTED_INPUT_SUFFIXES``.
    ValueError
        If the file exists and has a supported extension but OpenCV could
        not decode it (e.g. it is corrupt or not actually an image).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise UnsupportedFormatError(
            f"Unsupported file format {path.suffix!r} for {path}. Supported formats: {supported}"
        )

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image file: {path}")

    return image


def resize_to_max_dimension(image: np.ndarray, max_dimension: int | None) -> np.ndarray:
    """Downscale ``image`` so neither side exceeds ``max_dimension``.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W, ...)``.
    max_dimension : int | None
        Maximum allowed width/height in pixels. If ``None`` or the image
        already fits, the image is returned unchanged (no upscaling).

    Returns
    -------
    np.ndarray
        The original image, or a proportionally resized copy.
    """
    if max_dimension is None:
        return image

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dimension:
        return image

    scale = max_dimension / longest_side
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def derive_kernel_size(
    working_dimension: int, *, fraction: float, min_value: int = 3, odd: bool = True
) -> int:
    """Scale a pixel kernel size to a fraction of the working resolution.

    Engines used to hardcode kernel sizes (bilateral filter diameter,
    adaptive-threshold block size, morphology kernels, minimum stroke
    extent) as fixed pixel counts, which are only correct at whatever
    resolution they happened to be tuned at. Expressing them instead as
    a fraction of :data:`DEFAULT_WORKING_DIMENSION` lets every engine's
    kernels scale automatically with the working resolution.

    Parameters
    ----------
    working_dimension : int
        Longest side, in pixels, of the image the kernel will be applied
        to (typically the working-resolution image an engine receives).
    fraction : float
        Desired kernel size as a fraction of ``working_dimension``.
    min_value : int, optional
        Smallest allowed kernel size, by default 3 (below which most
        OpenCV filters degenerate or reject the kernel outright).
    odd : bool, optional
        If True (the default), round the result up to the nearest odd
        value, as required by filters like ``cv2.bilateralFilter`` and
        ``cv2.adaptiveThreshold`` whose kernel/block size must be odd.

    Returns
    -------
    int
        The derived kernel size, at least ``min_value``.
    """
    size = max(min_value, round(working_dimension * fraction))
    if odd and size % 2 == 0:
        size += 1
    return size


def default_line_thickness(working_dimension: int) -> int:
    """Compute a sensible default output line thickness for a given resolution.

    A 1px stroke is a hairline at typical print resolutions and is not
    practical to color inside; the reference output this project targets
    uses roughly 2-3px strokes at ~864px wide. Scaling thickness with
    the working resolution keeps that same visual weight regardless of
    ``--max-dimension``, instead of a fixed pixel count that reads as
    heavy at low resolution and vanishingly thin at high resolution.

    Parameters
    ----------
    working_dimension : int
        Longest side, in pixels, of the image being converted.

    Returns
    -------
    int
        Recommended line thickness in pixels, never less than 2.
    """
    return max(2, round(working_dimension / 500))


def smooth_preserving_edges(
    gray: np.ndarray,
    *,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Denoise a grayscale image while keeping strong edges sharp.

    A plain Gaussian/median blur softens real object edges along with
    sensor/texture noise, which is what produces jittery, broken outlines
    downstream. A bilateral filter blurs only pixels that are both close
    in space and similar in intensity, so it removes that noise without
    smearing genuine boundaries -- the standard first step before
    Canny/adaptive-threshold in photo-to-line-art pipelines.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``.
    d : int, optional
        Diameter of the pixel neighborhood used during filtering, by
        default 9.
    sigma_color : float, optional
        Filter sigma in color space; larger values mix more distant
        intensities together, by default 75.0.
    sigma_space : float, optional
        Filter sigma in coordinate space; larger values let farther
        pixels influence each other, by default 75.0.

    Returns
    -------
    np.ndarray
        Denoised image with the same shape and dtype as ``gray``.
    """
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def remove_short_strokes(binary_image: np.ndarray, *, min_extent: int = 4) -> np.ndarray:
    """Erase ink components that are too small to read as an intentional stroke.

    Raw edge/threshold output often contains single-pixel or few-pixel
    noise dots scattered across otherwise flat regions. These read as
    stray marks on a coloring page rather than intentional strokes, so
    small components are dropped (painted over with background) while
    real strokes are left untouched.

    Filtering is done on each component's *extent* (the longer of its
    bounding-box width/height) rather than its pixel area, because area
    is the wrong metric for strokes: a long, thin line can have an area
    as small as a stray dot, and would be wrongly erased by an area
    threshold even though it is clearly a real line. Extent instead
    tracks how far a component actually reaches across the page, which
    a genuine stroke does and a speck does not.

    Parameters
    ----------
    binary_image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background (paper) and darker pixels are ink, as
        produced by the conversion engines in this package.
    min_extent : int, optional
        Minimum bounding-box extent (the larger of width/height), in
        pixels, for a component to be kept, by default 4.

    Returns
    -------
    np.ndarray
        Copy of ``binary_image`` with small ink components erased to
        white.
    """
    ink_mask = (binary_image < 128).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)

    # Building a boolean "keep this label" lookup table and indexing the
    # whole label map with it in one vectorized pass avoids rescanning
    # the full image once per small component, which is what made the
    # previous loop-based implementation quadratic-ish on noisy images
    # with thousands of small components.
    extents = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = extents >= min_extent
    keep[0] = True  # background label; irrelevant since it's excluded by ink_mask below

    cleaned = np.where(keep[labels], binary_image, 255).astype(np.uint8)
    return cleaned


def remove_small_specks(binary_image: np.ndarray, *, min_area: int = 4) -> np.ndarray:
    """Deprecated alias for :func:`remove_short_strokes`.

    .. deprecated::
        Use :func:`remove_short_strokes` instead. ``min_area`` is passed
        through as ``min_extent``; the two are not numerically
        equivalent (area vs. bounding-box extent), so callers relying on
        precise area-based behavior should migrate explicitly rather
        than assume identical output.
    """
    warnings.warn(
        "remove_small_specks is deprecated; use remove_short_strokes "
        "(area filtering replaced by extent filtering) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return remove_short_strokes(binary_image, min_extent=min_area)


class DebugSink:
    """Writes an engine's intermediate conversion stages to disk.

    Normally only the final line-art image is ever written to disk, so
    when a result looks wrong there's no way to tell which internal
    stage (flattening, edge detection, redraw, ...) actually produced
    the bad output. Passing an instance of this class to
    :func:`convert_image` (via its ``debug_dir`` argument) lets engines
    save each stage they go through as a separate, numbered file.
    """

    def __init__(self, directory: Path) -> None:
        """Create a sink that writes into ``directory``, creating it if needed.

        Parameters
        ----------
        directory : Path
            Destination directory for saved debug images.
        """
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stage_count = 0

    def save(self, stage_name: str, image: np.ndarray) -> None:
        """Write one named intermediate image, prefixed with its stage order.

        Parameters
        ----------
        stage_name : str
            Short, filename-safe label for this stage (e.g. ``"flattened"``).
        image : np.ndarray
            The image to save at this stage.
        """
        self._stage_count += 1
        filename = f"{self._stage_count:02d}_{stage_name}.png"
        cv2.imwrite(str(self.directory / filename), image)


def convert_image(
    image: np.ndarray,
    engine: ConversionEngine,
    *,
    line_thickness: int = 1,
    max_dimension: int | None = DEFAULT_WORKING_DIMENSION,
    debug_dir: Path | None = None,
) -> np.ndarray:
    """Resize and run a single image through a conversion engine.

    Parameters
    ----------
    image : np.ndarray
        BGR input image, as returned by :func:`load_image`.
    engine : ConversionEngine
        The engine used to detect/render the line art.
    line_thickness : int, optional
        Approximate output line thickness in pixels, by default 1.
    max_dimension : int | None, optional
        The working resolution: the image is downscaled (never upscaled)
        so its longest side does not exceed this value before
        conversion, by default :data:`DEFAULT_WORKING_DIMENSION`. Every
        engine assumes it is operating at this scale when deriving its
        own resolution-dependent kernel sizes (see
        :func:`derive_kernel_size`); pass ``None`` to convert at the
        image's native resolution instead.
    debug_dir : Path | None, optional
        If given, a :class:`DebugSink` writing into this directory is
        passed to the engine, which may use it to save intermediate
        conversion stages for inspection, by default None (no debug
        output).

    Returns
    -------
    np.ndarray
        Grayscale line-art image, shape ``(H, W)``, dtype ``uint8``.
    """
    resized = resize_to_max_dimension(image, max_dimension)
    debug_sink = DebugSink(debug_dir) if debug_dir is not None else None
    return engine.convert(resized, line_thickness=line_thickness, debug=debug_sink)


def save_image(image: np.ndarray, path: Path) -> None:
    """Write a grayscale line-art image to disk, creating parent dirs.

    Parameters
    ----------
    image : np.ndarray
        Single-channel image to save.
    path : Path
        Destination file path. The file extension determines the encoding
        (e.g. ``.png``, ``.jpg``).

    Raises
    ------
    OSError
        If OpenCV fails to encode/write the image to ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write output image to: {path}")
