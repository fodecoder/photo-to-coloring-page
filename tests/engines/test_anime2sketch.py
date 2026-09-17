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


def test_local_weights_folder_is_preferred_over_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from coloring_page.engines.anime2sketch import _resolve_weights_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COLORING_PAGE_ANIME2SKETCH_WEIGHTS", raising=False)
    local_weights = tmp_path / "weights" / "anime2sketch.pth"
    local_weights.parent.mkdir()
    local_weights.write_bytes(b"not a real checkpoint")

    assert _resolve_weights_path(None) == Path("weights") / "anime2sketch.pth"


_HAS_REAL_WEIGHTS = (Path("weights") / "anime2sketch.pth").exists()
_SKIP_NO_WEIGHTS = pytest.mark.skipif(
    not _HAS_REAL_WEIGHTS,
    reason="requires a manually downloaded weights/anime2sketch.pth (not bundled)",
)


@_SKIP_NO_WEIGHTS
def test_real_weights_produce_grayscale_line_art(synthetic_photo: np.ndarray) -> None:
    """End-to-end check using the real, manually downloaded weights.

    Skipped everywhere the weights file isn't present (CI, other
    contributors' machines) -- this only runs where a developer has
    followed the README's manual download steps.
    """
    from coloring_page.engines.anime2sketch import Anime2SketchEngine

    engine = Anime2SketchEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8


@_SKIP_NO_WEIGHTS
def test_binarize_true_produces_pure_black_and_white(synthetic_photo: np.ndarray) -> None:
    """The default `binarize=True` should yield crisp ink, not soft shading."""
    from coloring_page.engines.anime2sketch import Anime2SketchEngine

    engine = Anime2SketchEngine(binarize=True)
    result = engine.convert(synthetic_photo)

    assert set(np.unique(result).tolist()) <= {0, 255}


@_SKIP_NO_WEIGHTS
def test_binarize_false_keeps_soft_grayscale_shading(synthetic_photo: np.ndarray) -> None:
    """With `binarize=False`, the raw pencil-shaded network output is kept."""
    from coloring_page.engines.anime2sketch import Anime2SketchEngine

    engine = Anime2SketchEngine(binarize=False)
    result = engine.convert(synthetic_photo)

    # A real photo run through the (unbinarized) network practically never
    # collapses to pure black/white everywhere -- some mid-gray shading
    # should survive.
    assert len(np.unique(result)) > 2


@_SKIP_NO_WEIGHTS
def test_gamma_correction_avoids_solid_black_regions() -> None:
    """A very dark synthetic image shouldn't collapse into a solid black blob.

    Regression test for the failure mode observed on the real dinosaur
    illustration: without gamma-lifting shadow detail before inference,
    large dark/shadowed source regions come out as a solid black smear
    rather than line art.
    """
    from coloring_page.engines.anime2sketch import Anime2SketchEngine

    dark_photo = np.full((64, 64, 3), 20, dtype=np.uint8)
    engine = Anime2SketchEngine(gamma=1.6, binarize=False)
    result = engine.convert(dark_photo)

    # The output shouldn't be (close to) entirely black.
    assert result.mean() > 50
