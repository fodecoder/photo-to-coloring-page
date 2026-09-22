"""Tests for the standalone raster detail-line conversion engine (lineart-raster).

Skipped entirely when `torch`/`controlnet_aux` (the `lineart_raster`
extra) aren't installed. Deliberately does NOT `importorskip("sam2")` --
this engine must be usable without it; a passing run of this file with
`sam2` absent is itself evidence of that. `LineArtRasterEngine.convert()`
is exercised with a duck-typed stub model injected via its constructor,
mirroring `tests/engines/test_lineart.py`'s `_StubDetailModel` pattern --
no real weights or GPU needed. The real-weights end-to-end test at the
bottom does, and is skipped without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("controlnet_aux")

from coloring_page.artwork import RasterArtwork  # noqa: E402
from coloring_page.engines.lineart_raster import LineArtRasterEngine  # noqa: E402
from coloring_page.exceptions import WeightsMissingError  # noqa: E402


class _StubModel:
    """Duck-types the detail-line model callable with a fixed response map."""

    def __init__(self, response: np.ndarray) -> None:
        self._response = response

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._response


def test_requires_serial_execution_is_true() -> None:
    assert LineArtRasterEngine.requires_serial_execution is True


def test_not_experimental() -> None:
    assert LineArtRasterEngine.experimental is False


def test_registered_without_sam2() -> None:
    from coloring_page.engines.registry import ENGINES

    # This test file only importorskips torch/controlnet_aux, deliberately
    # not sam2 -- so this asserts the engine is available regardless of
    # whether sam2 happens to be installed in this environment.
    assert "lineart-raster" in ENGINES


class TestConvert:
    def test_returns_raster_artwork_matching_input_shape(self) -> None:
        response = np.full((64, 64), 200, dtype=np.uint8)
        engine = LineArtRasterEngine(resolution=64, model=_StubModel(response))
        image = np.zeros((48, 96, 3), dtype=np.uint8)

        artwork = engine.convert(image)

        assert isinstance(artwork, RasterArtwork)
        assert artwork.image.shape == (48, 96)
        assert artwork.image.dtype == np.uint8
        assert artwork.aspect_ratio == pytest.approx(96 / 48)

    def test_three_channel_response_is_converted_to_grayscale(self) -> None:
        response = np.full((64, 64, 3), 100, dtype=np.uint8)
        engine = LineArtRasterEngine(resolution=64, model=_StubModel(response))
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        artwork = engine.convert(image)

        assert artwork.image.ndim == 2

    def test_corrects_detector_polarity_by_default(self) -> None:
        """The detector's raw white-line-on-black response must come out as ink-on-paper."""
        response = np.zeros((64, 64), dtype=np.uint8)
        response[32, :] = 255  # a bright "line" on an otherwise dark response
        engine = LineArtRasterEngine(resolution=64, model=_StubModel(response))
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        artwork = engine.convert(image)

        assert int(artwork.image[32, 0]) == 0  # the line is now ink (near-black)
        assert int(artwork.image[0, 0]) == 255  # the background is now paper (white)

    def test_identity_gamma_only_applies_the_polarity_correction(self) -> None:
        """gamma=1.0 is a no-op on top of the unconditional 255-response polarity fix."""
        response = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape(64, 64)
        engine = LineArtRasterEngine(resolution=64, contrast_gamma=1.0, model=_StubModel(response))
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        artwork = engine.convert(image)

        np.testing.assert_array_equal(artwork.image, 255 - response)

    def test_gamma_below_one_darkens_midtones(self) -> None:
        response = np.full((64, 64), 128, dtype=np.uint8)
        engine = LineArtRasterEngine(resolution=64, contrast_gamma=0.5, model=_StubModel(response))
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        artwork = engine.convert(image)

        # After the unconditional polarity correction (255-128=127) the
        # gamma curve pushes this ink-positive midtone darker still.
        assert int(artwork.image[0, 0]) < 127

    def test_invert_cancels_the_polarity_correction(self) -> None:
        """invert=True flips a second time, undoing the unconditional correction."""
        response = np.full((64, 64), 60, dtype=np.uint8)
        engine = LineArtRasterEngine(resolution=64, invert=True, model=_StubModel(response))
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        artwork = engine.convert(image)

        assert int(artwork.image[0, 0]) == 60

    def test_output_is_not_binarized(self) -> None:
        """The whole point of this engine: no thresholding, antialiasing survives."""
        rng = np.random.default_rng(0)
        response = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        engine = LineArtRasterEngine(resolution=64, model=_StubModel(response))
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        artwork = engine.convert(image)

        unique_values = np.unique(artwork.image)
        assert len(unique_values) > 2

    def test_debug_sink_receives_detail_response(self) -> None:
        saved: dict[str, np.ndarray] = {}

        class _Sink:
            def save(self, stage_name: str, image: np.ndarray) -> None:
                saved[stage_name] = image

        response = np.full((64, 64), 200, dtype=np.uint8)
        engine = LineArtRasterEngine(resolution=64, model=_StubModel(response))
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        engine.convert(image, debug=_Sink())

        assert "detail_response" in saved


def test_missing_weights_raises_clear_error(tmp_path: Path) -> None:
    engine = LineArtRasterEngine(weights_path=tmp_path / "missing-netG.pth")
    dummy_photo = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(WeightsMissingError, match="not found"):
        engine.convert(dummy_photo)


_HAS_REAL_WEIGHTS = (Path("weights") / "netG.pth").exists()
_SKIP_NO_WEIGHTS = pytest.mark.skipif(
    not _HAS_REAL_WEIGHTS, reason="requires manually downloaded weights/netG.pth (not bundled)"
)


@_SKIP_NO_WEIGHTS
def test_real_weights_produce_a_raster_artwork(synthetic_photo: np.ndarray) -> None:
    """End-to-end check using the real, manually downloaded weights."""
    engine = LineArtRasterEngine(resolution=256)
    artwork = engine.convert(synthetic_photo)

    assert isinstance(artwork, RasterArtwork)
    assert artwork.image.shape == synthetic_photo.shape[:2]
    assert artwork.aspect_ratio == pytest.approx(
        synthetic_photo.shape[1] / synthetic_photo.shape[0]
    )
