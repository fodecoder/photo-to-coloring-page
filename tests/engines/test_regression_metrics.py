"""Cross-engine regression test guarding against silent total failures.

The xdog (0.05% ink coverage, essentially a blank page) and adaptive
(26% ink coverage, illegible stipple) bugs fixed in this project were
never caught by any test: both produced output that "looked plausible"
as a diff -- correct shape, correct dtype, some black and white pixels
-- while being quantifiably useless as a coloring page. This
parametrizes over every registered engine and asserts each one's ink
coverage on a synthetic photo falls in a broad sanity band, deliberately
wide enough that it doesn't pin the tighter real-photo 5-8% target on a
tiny synthetic gradient, but tight enough to catch a silent collapse to
near-0% or near-100% ink.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coloring_page.drawing import rasterize
from coloring_page.engines.registry import ENGINES
from coloring_page.validate import ink_coverage

_WEIGHTS_REQUIRED_STYLES = {
    "anime2sketch": Path("weights") / "anime2sketch.pth",
    "informative_drawings": Path("weights") / "informative_drawings.pth",
    "gated": Path("weights") / "informative_drawings.pth",
}

#: gated deliberately discards every edge chain a pretrained network has no
#: confidence about, and a tiny synthetic gradient image is exactly the
#: kind of content the network was never trained to have an opinion on --
#: it legitimately gates everything out there, unlike the other engines'
#: bugs this test was written to catch (a threshold/formula error
#: producing a near-blank page on *real* content too). Covered instead by
#: tests/engines/test_gated.py's own real-weights sanity check.
_EXCLUDED_STYLES = {"gated"}


def _should_skip(style: str) -> str | None:
    weights_path = _WEIGHTS_REQUIRED_STYLES.get(style)
    if weights_path is not None and not weights_path.exists():
        return f"requires a manually downloaded {weights_path} (not bundled)"
    return None


@pytest.mark.parametrize("style", sorted(ENGINES.keys() - _EXCLUDED_STYLES))
def test_ink_coverage_in_sanity_band(style: str, synthetic_photo: np.ndarray) -> None:
    skip_reason = _should_skip(style)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    engine = ENGINES[style]()
    drawing = engine.convert(synthetic_photo)
    result = rasterize(drawing, long_side_px=max(synthetic_photo.shape[:2]))

    coverage = ink_coverage(result)
    assert 0.005 <= coverage <= 0.40, (
        f"{style} ink coverage {coverage:.4f} is outside the sanity band "
        "[0.005, 0.40] -- likely a silent total failure (near-blank page "
        "or near-solid ink), not a tuning issue."
    )


@pytest.mark.parametrize("style", sorted(ENGINES))
def test_engine_returns_valid_drawing(style: str, synthetic_photo: np.ndarray) -> None:
    """Every registered engine must return a well-formed Drawing.

    Complements test_ink_coverage_in_sanity_band above (a raster-quality
    check): this instead verifies the vector contract itself --
    non-empty paths, every point normalized to [0, 1], no degenerate
    (fewer-than-2-point) path -- regardless of what the traced content
    looks like once rasterized. Unlike that test, this runs on every
    engine including `gated`: Drawing validity is an unrelated axis from
    ink-coverage sanity, so `gated`'s legitimate on-tiny-synthetic-input
    zero-confidence gating doesn't exempt it from this contract check.
    """
    skip_reason = _should_skip(style)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    engine = ENGINES[style]()
    drawing = engine.convert(synthetic_photo)

    assert len(drawing.paths) > 0, f"{style} returned a Drawing with no paths."
    for path in drawing.paths:
        assert len(path.points) >= 2, f"{style} produced a path with fewer than 2 points."
        assert np.all(path.points >= 0.0) and np.all(path.points <= 1.0), (
            f"{style} produced a path with points outside [0, 1]."
        )
