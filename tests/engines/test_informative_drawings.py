"""Tests for the optional Informative Drawings engine.

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
    from coloring_page.engines._informative_drawings_arch import build_generator

    model = build_generator()
    model.eval()

    dummy_input = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 1, 256, 256)


def test_generator_output_is_sigmoid_bounded() -> None:
    from coloring_page.engines._informative_drawings_arch import build_generator

    model = build_generator()
    model.eval()

    dummy_input = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.min().item() >= 0.0
    assert output.max().item() <= 1.0


def test_missing_weights_raises_clear_error(tmp_path: Path) -> None:
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    engine = InformativeDrawingsEngine(weights_path=tmp_path / "missing-weights.pth")
    dummy_photo = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError, match="not found"):
        engine.convert(dummy_photo)


def test_registered_only_when_torch_available() -> None:
    from coloring_page.engines.registry import ENGINES

    # This test file only runs when torch is importable (see
    # `importorskip` above), so the engine must be registered.
    assert "informative_drawings" in ENGINES


def test_local_weights_folder_is_preferred_over_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from coloring_page.engines.informative_drawings import _resolve_weights_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COLORING_PAGE_INFORMATIVE_DRAWINGS_WEIGHTS", raising=False)
    local_weights = tmp_path / "weights" / "informative_drawings.pth"
    local_weights.parent.mkdir()
    local_weights.write_bytes(b"not a real checkpoint")

    assert _resolve_weights_path(None) == Path("weights") / "informative_drawings.pth"


_HAS_REAL_WEIGHTS = (Path("weights") / "informative_drawings.pth").exists()
_SKIP_NO_WEIGHTS = pytest.mark.skipif(
    not _HAS_REAL_WEIGHTS,
    reason="requires a manually downloaded weights/informative_drawings.pth (not bundled)",
)


@_SKIP_NO_WEIGHTS
def test_real_weights_produce_grayscale_line_art(synthetic_photo: np.ndarray) -> None:
    """End-to-end check using the real, manually downloaded weights.

    Skipped everywhere the weights file isn't present (CI, other
    contributors' machines) -- this only runs where a developer has
    followed the README's manual download steps.
    """
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    engine = InformativeDrawingsEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8
