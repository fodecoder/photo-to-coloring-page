"""Regression guard: ``RASTER_THRESHOLDS`` must accept a realistic raster line-art image.

A hands-on measurement against this repo showed the vector pipeline
(binarize -> thin -> trace) rejects a real, good, antialiased line-art
reference: ~4.7% full-black ink, ~10-12% intermediate gray. This test
synthesizes a raster that matches those statistics and asserts it passes
under :data:`~coloring_page.validate.RASTER_THRESHOLDS` -- a threshold
that rejects its own ideal is a broken threshold, and that has already
happened twice in this repo (see the module docstrings this test sits
next to).
"""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.artwork import RasterArtwork
from coloring_page.page import PageSpec
from coloring_page.validate import RASTER_THRESHOLDS, antialiased_fraction, ink_coverage, validate

_SPEC = PageSpec(size=(100.0, 100.0), margin_mm=5.0, dpi=150)


def _synthetic_lineart(size: int = 600) -> np.ndarray:
    """Antialiased strokes on white, tuned to land near ink~4-5%, midtones~10%.

    Draws non-crossing wavy strokes stacked in horizontal bands -- deliberately
    never intersecting, so blurring + the 1px morphological closing
    ``validate()`` applies to raster ink never stitches two strokes into a
    spurious tiny closed loop (a real hazard of random crossing strokes,
    and not something this calibration test is trying to exercise; enclosed-
    region area calibration is a separate concern). ``cv2.LINE_AA`` alone
    already produces soft, non-binary edges; a light Gaussian blur on top
    ensures a meaningful fraction of pixels sit strictly between black and
    white -- the stroke-width modulation a real detector output has and a
    binarized mask doesn't.
    """
    canvas = np.full((size, size), 255, dtype=np.uint8)

    rng = np.random.default_rng(0)
    xs = np.linspace(0, size - 1, 200)
    num_bands = 10
    band_height = size / (num_bands + 1)
    for i in range(num_bands):
        y0 = band_height * (i + 1)
        amplitude = rng.uniform(5.0, 12.0)
        freq = rng.uniform(1.5, 3.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        ys = y0 + amplitude * np.sin(freq * xs / size * 2 * np.pi + phase)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        thickness = int(rng.integers(2, 5))
        cv2.polylines(
            canvas, [pts], isClosed=False, color=0, thickness=thickness, lineType=cv2.LINE_AA
        )

    blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=1.2)
    return blurred


def _tune_to_target(image: np.ndarray, *, target_ink: float = 0.045) -> np.ndarray:
    """Gamma-adjust ``image`` so its measured ink coverage lands near ``target_ink``."""
    current = ink_coverage(image)
    if current <= 0:
        return image
    # A gamma curve on the *inverted* (ink-positive) signal shifts how much
    # of the midtone falls below the ink threshold without touching pure
    # white (255 stays 255) -- exactly the "only permitted manipulation"
    # convention this project's raster engines use elsewhere.
    inv = 255 - image.astype(np.float64)
    gamma = np.log(target_ink + 1e-6) / np.log(current + 1e-6)
    gamma = float(np.clip(gamma, 0.3, 3.0))
    adjusted_inv = 255.0 * (inv / 255.0) ** gamma
    return (255.0 - adjusted_inv).clip(0, 255).astype(np.uint8)


class TestRasterCalibration:
    def test_synthetic_lineart_lands_in_target_stat_band(self) -> None:
        """The generator itself must actually hit the measured real-world stats."""
        image = _tune_to_target(_synthetic_lineart())
        assert 0.03 <= ink_coverage(image) <= 0.08
        assert antialiased_fraction(image) > 0.05

    def test_synthetic_lineart_passes_raster_validation(self) -> None:
        image = _tune_to_target(_synthetic_lineart())
        artwork = RasterArtwork(image=image, aspect_ratio=1.0)
        report = validate(artwork, _SPEC, **RASTER_THRESHOLDS)
        assert report.passed is True

    def test_leaking_and_dangling_are_not_applicable_for_raster(self) -> None:
        image = _tune_to_target(_synthetic_lineart())
        artwork = RasterArtwork(image=image, aspect_ratio=1.0)
        report = validate(artwork, _SPEC)
        assert report.leaking_regions is None
        assert report.dangling_endpoints is None

    def test_binarized_raster_fails_antialiased_fraction(self) -> None:
        """A silently-binarized raster must be caught, not accepted as 'good'."""
        image = _tune_to_target(_synthetic_lineart())
        _, binarized = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        artwork = RasterArtwork(image=binarized, aspect_ratio=1.0)
        report = validate(artwork, _SPEC, **RASTER_THRESHOLDS)
        # Page-placement resampling (to_png's INTER_AREA/LANCZOS4 resize)
        # reintroduces a thin sliver of intermediate gray at stroke edges
        # even from fully binary input, so this can't assert exactly 0 --
        # only that it stays far below the "real antialiasing" band a
        # non-binarized engine output lands in.
        assert report.antialiased_fraction is not None
        assert report.antialiased_fraction < 0.05  # RASTER_THRESHOLDS["min_antialiased_fraction"]
        assert report.passed is False
