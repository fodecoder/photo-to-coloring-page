"""Tests for the SAM-segmentation + detail-line (lineart) conversion engine.

Skipped entirely when `torch`/`sam2`/`controlnet_aux` (the `lineart`
extra) aren't installed. Every stage function is tested independently on
synthetic arrays with no model involved; `LineArtEngine.convert()` itself
is exercised end-to-end with duck-typed stub models injected via its
constructor, mirroring tests/engines/test_gated.py's
`_StubInformativeDrawings` pattern -- none of this needs real weights or a
GPU. The real-weights end-to-end test at the bottom does, and is skipped
without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from coloring_page.drawing import Path as DrawingPath
from coloring_page.exceptions import WeightsMissingError

torch = pytest.importorskip("torch")
pytest.importorskip("sam2")
pytest.importorskip("controlnet_aux")

from coloring_page.engines.lineart import (  # noqa: E402
    LineArtEngine,
    _adjacent_labels,
    _chaikin_smooth,
    _drop_hausdorff_duplicates,
    _merge_and_prune,
    _mm2_to_px2,
    _mm_to_px,
    _regularize,
)


def test_requires_serial_execution_is_true() -> None:
    assert LineArtEngine.requires_serial_execution is True


class _StubMaskGenerator:
    """Duck-types SAM2AutomaticMaskGenerator's `.generate` with fixed masks."""

    def __init__(self, masks: list[dict[str, Any]]) -> None:
        self._masks = masks

    def generate(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        return self._masks


class _StubDetailModel:
    """Duck-types the detail-line model callable with a fixed response map."""

    def __init__(self, response: np.ndarray) -> None:
        self._response = response

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._response


def _two_region_masks(shape: tuple[int, int]) -> list[dict[str, Any]]:
    """One large region and one small region, covering the whole image."""
    height, width = shape
    small = np.zeros(shape, dtype=bool)
    small[: height // 4, : width // 4] = True
    large = ~small
    return [
        {"segmentation": small, "area": int(small.sum())},
        {"segmentation": large, "area": int(large.sum())},
    ]


def test_registered_only_when_extras_available() -> None:
    from coloring_page.engines.registry import ENGINES

    # This test file only runs when torch/sam2/controlnet_aux are all
    # importable (see `importorskip` above), so the engine must be
    # registered.
    assert "lineart" in ENGINES


def test_mm_to_px_exact_arithmetic() -> None:
    assert _mm_to_px(25.4, dpi=300.0) == pytest.approx(300.0)
    assert _mm_to_px(0.0, dpi=300.0) == 0.0


def test_mm2_to_px2_exact_arithmetic() -> None:
    # 1 inch^2 = 25.4mm^2 side, so at 300dpi that's 300x300 px^2.
    assert _mm2_to_px2(25.4 * 25.4, dpi=300.0) == pytest.approx(300.0 * 300.0)


def test_chaikin_smooth_cuts_corners_of_a_square() -> None:
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)

    smoothed = _chaikin_smooth(square, iterations=1, closed=True)

    assert len(smoothed) == len(square) * 2
    # No original corner survives after one closed-path iteration.
    for corner in square:
        assert not any(np.allclose(corner, point) for point in smoothed)


def test_chaikin_smooth_preserves_open_path_endpoints() -> None:
    line = np.array([[0, 0], [5, 5], [10, 0]], dtype=np.float32)

    smoothed = _chaikin_smooth(line, iterations=2, closed=False)

    np.testing.assert_array_equal(smoothed[0], line[0])
    np.testing.assert_array_equal(smoothed[-1], line[-1])


def test_chaikin_smooth_zero_iterations_is_identity() -> None:
    line = np.array([[0, 0], [5, 5], [10, 0]], dtype=np.float32)
    result = _chaikin_smooth(line, iterations=0, closed=False)
    np.testing.assert_array_equal(result, line)


def test_adjacent_labels_finds_both_sides_of_a_seam() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[:, 5:] = 1
    # A vertical contour segment straddling the seam.
    contour = np.array([[[4, y]] for y in range(10)], dtype=np.int32)

    result = _adjacent_labels(contour, labels, 10, 10)

    assert result == (0, 1)


def test_adjacent_labels_returns_none_with_only_one_label() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    contour = np.array([[[4, 4]]], dtype=np.int32)

    assert _adjacent_labels(contour, labels, 10, 10) is None


def test_merge_and_prune_keeps_boundary_on_area_alone() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1  # two large, same-color-ish regions
    image = np.full((40, 40, 3), 128, dtype=np.uint8)
    detail_response = np.full((40, 40), 255, dtype=np.uint8)  # no detail ink anywhere

    chains, detail_mask = _merge_and_prune(
        labels,
        detail_response,
        image,
        min_region_area_px2=1.0,  # both regions are far larger than this
        min_region_contrast=1000.0,  # contrast alone would never pass
        detail_threshold=1.0,
    )

    assert len(chains) >= 1
    assert not detail_mask.any()


def test_merge_and_prune_drops_tiny_low_contrast_boundary() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:2, :2] = 1  # a tiny 2x2 region against a large, same-color one
    image = np.full((40, 40, 3), 128, dtype=np.uint8)
    detail_response = np.full((40, 40), 255, dtype=np.uint8)

    chains, _ = _merge_and_prune(
        labels,
        detail_response,
        image,
        min_region_area_px2=1_000_000.0,  # far larger than either region
        min_region_contrast=1000.0,  # same color everywhere, never passes
        detail_threshold=1.0,
    )

    assert chains == []


def test_merge_and_prune_drops_detail_ink_inside_a_discarded_region() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:4, :4] = 1  # a tiny region, discarded by both criteria below
    image = np.full((40, 40, 3), 128, dtype=np.uint8)
    detail_response = np.full((40, 40), 255, dtype=np.uint8)
    detail_response[:4, :4] = 0  # strong detail ink, entirely inside label 1

    _, detail_mask = _merge_and_prune(
        labels,
        detail_response,
        image,
        min_region_area_px2=1_000_000.0,
        min_region_contrast=1000.0,
        detail_threshold=128.0,
    )

    assert not detail_mask[:4, :4].any()


def test_merge_and_prune_keeps_detail_ink_inside_a_kept_region() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, 20:] = 255  # high-contrast seam -> boundary survives
    detail_response = np.full((40, 40), 255, dtype=np.uint8)
    detail_response[5:8, 5:8] = 0  # strong detail ink inside the kept region

    _, detail_mask = _merge_and_prune(
        labels,
        detail_response,
        image,
        min_region_area_px2=1_000_000.0,
        min_region_contrast=10.0,
        detail_threshold=128.0,
    )

    assert detail_mask[5:8, 5:8].any()


def test_drop_hausdorff_duplicates_collapses_near_identical_paths() -> None:
    boundary = DrawingPath(
        points=np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]], dtype=np.float32),
        closed=False,
        kind="boundary",
    )
    near_duplicate_detail = DrawingPath(
        points=np.array([[0.101, 0.1], [0.5, 0.101], [0.501, 0.5]], dtype=np.float32),
        closed=False,
        kind="detail",
    )

    result = _drop_hausdorff_duplicates((boundary, near_duplicate_detail), threshold=0.01)

    assert len(result) == 1
    assert result[0].kind == "boundary"


def test_drop_hausdorff_duplicates_keeps_genuinely_different_paths() -> None:
    boundary = DrawingPath(
        points=np.array([[0.1, 0.1], [0.5, 0.1]], dtype=np.float32), closed=False, kind="boundary"
    )
    unrelated_detail = DrawingPath(
        points=np.array([[0.8, 0.8], [0.9, 0.9]], dtype=np.float32), closed=False, kind="detail"
    )

    result = _drop_hausdorff_duplicates((boundary, unrelated_detail), threshold=0.01)

    assert len(result) == 2


def test_regularize_prunes_short_paths() -> None:
    tiny_chain = np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]], dtype=np.int32)
    empty_detail_mask = np.zeros((100, 100), dtype=np.uint8)

    drawing = _regularize(
        [tiny_chain],
        empty_detail_mask,
        (100, 100),
        1.0,
        polyline_epsilon_px=1.2,
        min_path_length_px=1000.0,  # far longer than this tiny chain
        hausdorff_threshold_px=1.0,
        chaikin_iterations=1,
    )

    assert drawing.paths == ()


def test_regularize_keeps_a_long_enough_path() -> None:
    big_chain = np.array([[[0, 0]], [[0, 50]], [[50, 50]], [[50, 0]]], dtype=np.int32)
    empty_detail_mask = np.zeros((100, 100), dtype=np.uint8)

    drawing = _regularize(
        [big_chain],
        empty_detail_mask,
        (100, 100),
        1.0,
        polyline_epsilon_px=1.2,
        min_path_length_px=1.0,
        hausdorff_threshold_px=1.0,
        chaikin_iterations=1,
    )

    assert len(drawing.paths) == 1
    assert drawing.paths[0].kind == "boundary"


def test_convert_end_to_end_with_stub_models(synthetic_photo: np.ndarray) -> None:
    height, width = synthetic_photo.shape[:2]
    masks = _two_region_masks((height, width))
    detail_response = np.full((height, width), 255, dtype=np.uint8)

    engine = LineArtEngine(
        mask_generator=_StubMaskGenerator(masks),
        detail_model=_StubDetailModel(detail_response),
        min_region_area_mm2=0.001,
        min_region_contrast=0.0,
        min_path_length_mm=0.001,
        print_dpi=300.0,
    )

    drawing = engine.convert(synthetic_photo)

    assert drawing.aspect_ratio == width / height
    for path in drawing.paths:
        assert len(path.points) >= 2


def test_missing_segmentation_weights_raises_clear_error(tmp_path: Path) -> None:
    engine = LineArtEngine(
        weights_path=tmp_path / "missing-sam2.pt",
        detail_model=_StubDetailModel(np.full((16, 16), 255, dtype=np.uint8)),
    )
    dummy_photo = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(WeightsMissingError, match="not found"):
        engine.convert(dummy_photo)


def test_missing_detail_weights_raises_clear_error(tmp_path: Path) -> None:
    engine = LineArtEngine(
        mask_generator=_StubMaskGenerator(_two_region_masks((16, 16))),
        detail_weights_path=tmp_path / "missing-netG.pth",
    )
    dummy_photo = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(WeightsMissingError, match="not found"):
        engine.convert(dummy_photo)


_HAS_REAL_WEIGHTS = (Path("weights") / "sam2.1_hiera_small.pt").exists() and (
    Path("weights") / "netG.pth"
).exists()
_SKIP_NO_WEIGHTS = pytest.mark.skipif(
    not _HAS_REAL_WEIGHTS,
    reason="requires manually downloaded weights/sam2.1_hiera_small.pt and weights/netG.pth "
    "(not bundled)",
)


@_SKIP_NO_WEIGHTS
def test_real_weights_produce_a_drawing(synthetic_photo: np.ndarray) -> None:
    """End-to-end check using the real, manually downloaded weights."""
    engine = LineArtEngine()
    drawing = engine.convert(synthetic_photo)

    assert drawing.aspect_ratio == synthetic_photo.shape[1] / synthetic_photo.shape[0]
    for path in drawing.paths:
        assert len(path.points) >= 2
