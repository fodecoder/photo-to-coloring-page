"""Shared image I/O and pre/post-processing used by all conversion engines."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from coloring_page.artwork import Artwork
from coloring_page.engines.base import ConversionEngine
from coloring_page.exceptions import UnsupportedImageError

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


#: Compatibility alias for code written against this package before typed
#: errors existed. New code should catch ``UnsupportedImageError`` (or its
#: base, ``ColoringPageError``) directly.
UnsupportedFormatError = UnsupportedImageError


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
    UnsupportedImageError
        If ``path``'s extension is not one of ``SUPPORTED_INPUT_SUFFIXES``,
        or the file exists and has a supported extension but OpenCV could
        not decode it (e.g. it is corrupt or not actually an image).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise UnsupportedImageError(
            f"Unsupported file format {path.suffix!r} for {path}. Supported formats: {supported}"
        )

    image = cv2.imread(str(path))
    if image is None:
        raise UnsupportedImageError(f"Could not decode image file: {path}")

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

    Measured against the project's reference output (``docs/desired.jpg``,
    4.37% ink coverage at 864px wide), the previous ``working_dimension /
    500`` formula produced strokes roughly 3x too thick at typical working
    resolutions (~3px at 1400px) without any measurable boundary F1 gain
    over thinner strokes -- it was inflating ink coverage, not quality.
    This targets ~0.15% of the long side, i.e. roughly 1-2px in the
    864-1400px range engines are actually tuned at, while still scaling
    with ``--max-dimension`` so thickness keeps the same visual weight
    regardless of working resolution.

    Parameters
    ----------
    working_dimension : int
        Longest side, in pixels, of the image being converted.

    Returns
    -------
    int
        Recommended line thickness in pixels, never less than 1.
    """
    return max(1, round(working_dimension / 700))


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


def run_pipeline(
    image: np.ndarray,
    engine: ConversionEngine,
    *,
    max_dimension: int | None = DEFAULT_WORKING_DIMENSION,
    debug_dir: Path | None = None,
) -> Artwork:
    """Resize and run a single image through a conversion engine.

    Low-level primitive: takes an already-selected ``engine`` and returns
    its raw output (a ``Drawing`` or a ``RasterArtwork``, see
    :mod:`coloring_page.artwork`) with no page placement, detail
    filtering, or quality validation applied.
    :func:`coloring_page.api.convert_image` is the public,
    ``Profile``-driven entrypoint built on top of this.

    Parameters
    ----------
    image : np.ndarray
        BGR input image, as returned by :func:`load_image`.
    engine : ConversionEngine
        The engine used to detect/trace the line art.
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
    Artwork
        Either a ``Drawing`` (normalized vector line art) or a
        ``RasterArtwork`` -- see :mod:`coloring_page.artwork`.
    """
    resized = resize_to_max_dimension(image, max_dimension)
    debug_sink = DebugSink(debug_dir) if debug_dir is not None else None
    return engine.convert(resized, debug=debug_sink)


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
