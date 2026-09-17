"""Tests for the optional Anime2Sketch engine.

Skipped entirely when `torch` (the `ml` extra) isn't installed. No
pretrained weights are bundled with this project, so these tests cover
the network architecture and the missing-weights error path rather than
requiring an actual weights file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_generator_forward_pass_produces_expected_shape() -> None:
    from coloring_page.engines._anime2sketch_arch import build_generator

    model = build_generator()
    model.eval()

    # 256x256 is the minimum input size compatible with the network's 8
    # downsampling stages (2**8 == 256).
    dummy_input = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 1, 256, 256)


def test_missing_weights_raises_clear_error(tmp_path: Path) -> None:
    from coloring_page.engines.anime2sketch import Anime2SketchEngine

    engine = Anime2SketchEngine(weights_path=tmp_path / "missing-weights.pth")
    dummy_photo = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError, match="not found"):
        engine.convert(dummy_photo)


def test_registered_only_when_torch_available() -> None:
    from coloring_page.engines.registry import ENGINES

    # This test file only runs when torch is importable (see
    # `importorskip` above), so the engine must be registered.
    assert "anime2sketch" in ENGINES
