"""Shared image I/O and pre/post-processing used by all conversion engines."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine

#: File extensions accepted as input photos.
SUPPORTED_INPUT_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


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


def remove_small_specks(binary_image: np.ndarray, *, min_area: int = 4) -> np.ndarray:
    """Erase isolated ink specks that are too small to be a real line.

    Raw edge/threshold output often contains single-pixel or few-pixel
    noise dots scattered across otherwise flat regions. These read as
    stray marks on a coloring page rather than intentional strokes, so
    connected components below ``min_area`` are dropped (painted over
    with background) while larger strokes are left untouched.

    Parameters
    ----------
    binary_image : np.ndarray
        Single-channel image, shape ``(H, W)``, dtype ``uint8``, where
        ``255`` is background (paper) and darker pixels are ink, as
        produced by the conversion engines in this package.
    min_area : int, optional
        Minimum connected-component size, in pixels, for a component to
        be kept, by default 4.

    Returns
    -------
    np.ndarray
        Copy of ``binary_image`` with small ink specks erased to white.
    """
    ink_mask = (binary_image < 128).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)

    cleaned = binary_image.copy()
    for label in range(1, num_labels):  # label 0 is the background
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == label] = 255

    return cleaned


def convert_image(
    image: np.ndarray,
    engine: ConversionEngine,
    *,
    line_thickness: int = 1,
    max_dimension: int | None = None,
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
        If set, the image is downscaled (never upscaled) so its longest
        side does not exceed this value before conversion, by default None.

    Returns
    -------
    np.ndarray
        Grayscale line-art image, shape ``(H, W)``, dtype ``uint8``.
    """
    resized = resize_to_max_dimension(image, max_dimension)
    return engine.convert(resized, line_thickness=line_thickness)


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
