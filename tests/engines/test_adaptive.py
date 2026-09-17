"""Tests for the adaptive-threshold conversion engine."""

from __future__ import annotations

import numpy as np

from coloring_page.engines.adaptive import AdaptiveEngine


def test_output_shape_and_dtype(synthetic_photo: np.ndarray) -> None:
    engine = AdaptiveEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


def test_output_is_binary_line_art(synthetic_photo: np.ndarray) -> None:
    engine = AdaptiveEngine()
    result = engine.convert(synthetic_photo)

    assert set(np.unique(result).tolist()) <= {0, 255}


def test_even_block_size_is_made_odd() -> None:
    engine = AdaptiveEngine(block_size=10)
    assert engine.block_size == 11
