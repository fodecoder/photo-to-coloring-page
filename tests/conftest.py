"""Shared pytest fixtures for the coloring-page test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def make_synthetic_photo(width: int = 64, height: int = 48) -> np.ndarray:
    """Build a small synthetic BGR "photo" with clear shapes and gradients.

    Generated in-memory (no binary fixture needed) so it has enough
    structure -- edges, gradients, flat regions -- to exercise every
    conversion engine meaningfully.

    Parameters
    ----------
    width : int, optional
        Image width in pixels, by default 64.
    height : int, optional
        Image height in pixels, by default 48.

    Returns
    -------
    np.ndarray
        BGR image array with shape ``(height, width, 3)`` and dtype ``uint8``.
    """
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    gradient = (x_grid * 255 / max(width - 1, 1)).astype(np.uint8)

    image = np.stack([gradient, gradient, gradient], axis=-1).astype(np.uint8)

    # A solid rectangle and a circle give the engines hard edges to find.
    image[height // 4 : height // 2, width // 4 : width // 2] = (10, 10, 10)
    cy, cx, radius = height * 3 // 4, width * 3 // 4, min(height, width) // 6
    yy, xx = np.ogrid[:height, :width]
    circle_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    image[circle_mask] = (240, 240, 240)

    return image


@pytest.fixture
def synthetic_photo() -> np.ndarray:
    """A small synthetic BGR photo used to exercise conversion engines."""
    return make_synthetic_photo()


@pytest.fixture
def synthetic_photo_path(tmp_path: Path, synthetic_photo: np.ndarray) -> Path:
    """Write a synthetic photo to a temporary PNG file and return its path."""
    path = tmp_path / "photo.png"
    # PIL expects RGB; the synthetic image is symmetric across channels
    # (grayscale gradient plus neutral shapes), so BGR vs RGB is equivalent.
    Image.fromarray(synthetic_photo).save(path)
    return path
