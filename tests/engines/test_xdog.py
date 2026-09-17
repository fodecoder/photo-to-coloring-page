"""Tests for the XDoG conversion engine."""

from __future__ import annotations

import numpy as np

from coloring_page.engines.xdog import XDoGEngine


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = XDoGEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_has_light_background(synthetic_photo: np.ndarray) -> None:
    engine = XDoGEngine()
    result = engine.convert(synthetic_photo)

    # Most of the synthetic photo is flat/gradient area with no edges,
    # so the majority of output pixels should be near-white background.
    assert np.mean(result) > 127
