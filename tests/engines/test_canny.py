"""Tests for the Canny conversion engine."""

from __future__ import annotations

import numpy as np

from coloring_page.engines.canny import CannyEngine


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_is_binary_line_art(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    result = engine.convert(synthetic_photo)

    # Canny output is strictly black-or-white; inverted it stays that way.
    assert set(np.unique(result).tolist()) <= {0, 255}


def test_higher_thickness_adds_more_ink(synthetic_photo: np.ndarray) -> None:
    engine = CannyEngine()
    thin = engine.convert(synthetic_photo, line_thickness=1)
    thick = engine.convert(synthetic_photo, line_thickness=3)

    thin_ink = np.sum(thin < 128)
    thick_ink = np.sum(thick < 128)
    assert thick_ink >= thin_ink
