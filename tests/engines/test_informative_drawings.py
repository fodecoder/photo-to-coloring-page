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

from coloring_page.exceptions import WeightsMissingError

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


def test_postprocess_strategy_defaults_to_hysteresis() -> None:
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    assert InformativeDrawingsEngine().postprocess_strategy == "hysteresis"


def test_missing_weights_raises_clear_error(tmp_path: Path) -> None:
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    engine = InformativeDrawingsEngine(weights_path=tmp_path / "missing-weights.pth")
    dummy_photo = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(WeightsMissingError, match="not found"):
        engine.convert(dummy_photo)


def test_mismatched_checkpoint_reports_missing_keys(tmp_path: Path) -> None:
    from coloring_page.engines._informative_drawings_arch import build_generator
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    # A real state dict with one key removed: strict loading should name
    # exactly which key is missing, not just fail generically.
    state_dict = build_generator().state_dict()
    del state_dict[next(iter(state_dict))]

    weights_path = tmp_path / "corrupted.pth"
    torch.save(state_dict, weights_path)

    engine = InformativeDrawingsEngine(weights_path=weights_path)
    dummy_photo = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="missing_keys"):
        engine.convert(dummy_photo)


def test_resize_short_side_targets_short_side_and_rounds_to_multiple() -> None:
    from coloring_page.engines.informative_drawings import _resize_short_side

    image = np.zeros((300, 600, 3), dtype=np.uint8)  # short side = 300 (height)
    resized = _resize_short_side(image, 512, multiple=64)

    height, width = resized.shape[:2]
    assert height % 64 == 0
    assert width % 64 == 0
    # Short side (height) should land near the requested 512, not the
    # long side -- squashing to a square was the bug this fixes.
    assert abs(height - 512) < 64
    assert width > height  # aspect ratio roughly preserved, not squared off


def test_registered_only_when_torch_available() -> None:
    from coloring_page.engines.registry import ENGINES

    # This test file only runs when torch is importable (see
    # `importorskip` above), so the engine must be registered.
    assert "informative_drawings" in ENGINES


def test_requires_serial_execution_is_true() -> None:
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    assert InformativeDrawingsEngine.requires_serial_execution is True


def test_local_weights_folder_is_preferred_over_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from coloring_page.engines.informative_drawings import _resolve_weights_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COLORING_PAGE_INFORMATIVE_DRAWINGS_WEIGHTS", raising=False)
    monkeypatch.delenv("COLORING_PAGE_WEIGHTS_DIR", raising=False)
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
def test_soft_map_polarity_is_mostly_background(synthetic_photo: np.ndarray) -> None:
    """The network's raw output must be mostly background, not mostly ink.

    ``soft_map``'s docstring asserts the generator's Sigmoid output needs
    no inversion (0 = ink, 1 = background, same as this project's
    convention). A real photo's line-art extraction should be dominated
    by background -- if that polarity assumption were ever wrong (e.g. a
    checkpoint or architecture mismatch flipping the sign), the output
    would be mostly dark instead, which this catches directly rather than
    relying on the comment being correct.
    """
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    engine = InformativeDrawingsEngine()
    sketch = engine.soft_map(synthetic_photo)

    assert sketch.mean() > 128


@_SKIP_NO_WEIGHTS
def test_real_weights_produce_grayscale_line_art(synthetic_photo: np.ndarray) -> None:
    """End-to-end check using the real, manually downloaded weights.

    Skipped everywhere the weights file isn't present (CI, other
    contributors' machines) -- this only runs where a developer has
    followed the README's manual download steps.
    """
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    engine = InformativeDrawingsEngine()
    drawing = engine.convert(synthetic_photo)

    assert len(drawing.paths) > 0
    for path in drawing.paths:
        assert len(path.points) >= 2
