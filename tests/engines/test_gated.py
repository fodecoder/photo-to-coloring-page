"""Tests for the semantic-gated (chained + informative_drawings) engine.

Skipped entirely when `torch` (the `ml` extra) isn't installed. Gating
logic itself is tested with a stub network (anything with a `soft_map`
method) so most of this doesn't need real pretrained weights; the
end-to-end test at the bottom does, and is skipped without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _StubInformativeDrawings:
    """Duck-types InformativeDrawingsEngine's `soft_map` with a fixed return value."""

    def __init__(self, soft_map_value: np.ndarray) -> None:
        self._soft_map_value = soft_map_value

    def soft_map(self, image: np.ndarray) -> np.ndarray:
        return self._soft_map_value


def test_registered_only_when_torch_available() -> None:
    from coloring_page.engines.registry import ENGINES

    # This test file only runs when torch is importable (see
    # `importorskip` above), so the engine must be registered.
    assert "gated" in ENGINES


def test_segment_confidence_prefers_dark_regions() -> None:
    from coloring_page.engines.gated import GatedEngine

    engine = GatedEngine(informative_drawings=_StubInformativeDrawings(np.zeros((1, 1))))

    normalized_soft = np.full((30, 30), 255, dtype=np.uint8)
    normalized_soft[10, :] = 0  # a fully-confident (maximally ink-like) row

    segment_on_dark = np.array([[[x, 10]] for x in range(5, 25)], dtype=np.int32)
    segment_on_light = np.array([[[x, 20]] for x in range(5, 25)], dtype=np.int32)

    dark_confidence = engine._segment_confidence(segment_on_dark, normalized_soft)
    light_confidence = engine._segment_confidence(segment_on_light, normalized_soft)

    assert dark_confidence > light_confidence


def test_segment_confidence_on_empty_segment_is_zero() -> None:
    from coloring_page.engines.gated import GatedEngine

    engine = GatedEngine(
        informative_drawings=_StubInformativeDrawings(np.zeros((1, 1))), tolerance_radius=0
    )
    normalized_soft = np.full((10, 10), 255, dtype=np.uint8)
    empty_segment = np.zeros((0, 1, 2), dtype=np.int32)

    assert engine._segment_confidence(empty_segment, normalized_soft) == 0.0


def test_stricter_gate_threshold_keeps_no_more_ink(synthetic_photo: np.ndarray) -> None:
    from coloring_page.engines.gated import GatedEngine

    height, width = synthetic_photo.shape[:2]
    # A clean bimodal soft map (real percentile spread, no degenerate
    # normalization): left half maximally ink-like, right half background.
    soft = np.full((height, width), 255, dtype=np.uint8)
    soft[:, : width // 2] = 0

    permissive = GatedEngine(
        gate_threshold=1.0, informative_drawings=_StubInformativeDrawings(soft)
    )
    strict = GatedEngine(gate_threshold=200.0, informative_drawings=_StubInformativeDrawings(soft))

    permissive_result = permissive.convert(synthetic_photo)
    strict_result = strict.convert(synthetic_photo)

    assert np.sum(strict_result < 128) <= np.sum(permissive_result < 128)


def test_missing_weights_raises_clear_error(tmp_path: Path) -> None:
    from coloring_page.engines.gated import GatedEngine
    from coloring_page.engines.informative_drawings import InformativeDrawingsEngine

    engine = GatedEngine(
        informative_drawings=InformativeDrawingsEngine(weights_path=tmp_path / "missing.pth")
    )
    dummy_photo = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError, match="not found"):
        engine.convert(dummy_photo)


_HAS_REAL_WEIGHTS = (Path("weights") / "informative_drawings.pth").exists()
_SKIP_NO_WEIGHTS = pytest.mark.skipif(
    not _HAS_REAL_WEIGHTS,
    reason="requires a manually downloaded weights/informative_drawings.pth (not bundled)",
)


@_SKIP_NO_WEIGHTS
def test_real_weights_produce_grayscale_line_art(synthetic_photo: np.ndarray) -> None:
    """End-to-end check using the real, manually downloaded weights.

    Doesn't assert on ink coverage: gating can legitimately discard
    every chain on content (like this tiny synthetic gradient) the
    network has no real confidence about -- see
    tests/engines/test_regression_metrics.py's exclusion of "gated" for
    the same reason. Only shape/dtype are checked here.
    """
    from coloring_page.engines.gated import GatedEngine

    engine = GatedEngine()
    result = engine.convert(synthetic_photo)

    assert result.shape == synthetic_photo.shape[:2]
    assert result.dtype == np.uint8
