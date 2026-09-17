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


def add_salt_and_pepper_noise(
    image: np.ndarray, *, amount: float = 0.02, seed: int = 0
) -> np.ndarray:
    """Return a copy of ``image`` with isolated extreme-value noise pixels.

    Used to verify that engines' speckle-cleanup step drops noise dots
    that don't correspond to real object boundaries, rather than to
    exercise realistic sensor noise.

    Parameters
    ----------
    image : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
    amount : float, optional
        Fraction of pixels to replace with black or white noise, by
        default 0.02.
    seed : int, optional
        Seed for the noise pattern, for reproducible tests, by default 0.

    Returns
    -------
    np.ndarray
        Noisy copy of ``image``.
    """
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    height, width = image.shape[:2]
    num_noisy = int(amount * height * width)

    ys = rng.integers(0, height, size=num_noisy)
    xs = rng.integers(0, width, size=num_noisy)
    values = rng.choice([0, 255], size=num_noisy)
    noisy[ys, xs] = values[:, None]

    return noisy


@pytest.fixture
def synthetic_photo() -> np.ndarray:
    """A small synthetic BGR photo used to exercise conversion engines."""
    return make_synthetic_photo()


@pytest.fixture
def noisy_synthetic_photo(synthetic_photo: np.ndarray) -> np.ndarray:
    """A synthetic photo with isolated salt-and-pepper noise pixels added."""
    return add_salt_and_pepper_noise(synthetic_photo)


@pytest.fixture
def synthetic_photo_path(tmp_path: Path, synthetic_photo: np.ndarray) -> Path:
    """Write a synthetic photo to a temporary PNG file and return its path."""
    path = tmp_path / "photo.png"
    # PIL expects RGB; the synthetic image is symmetric across channels
    # (grayscale gradient plus neutral shapes), so BGR vs RGB is equivalent.
    Image.fromarray(synthetic_photo).save(path)
    return path
