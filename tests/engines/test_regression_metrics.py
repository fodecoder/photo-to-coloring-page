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

from coloring_page.engines.registry import ENGINES
from coloring_page.metrics import ink_coverage

_HAS_ANIME2SKETCH_WEIGHTS = (Path("weights") / "anime2sketch.pth").exists()


def _should_skip(style: str) -> str | None:
    if style == "anime2sketch" and not _HAS_ANIME2SKETCH_WEIGHTS:
        return "requires a manually downloaded weights/anime2sketch.pth (not bundled)"
    return None


@pytest.mark.parametrize("style", sorted(ENGINES))
def test_ink_coverage_in_sanity_band(style: str, synthetic_photo: np.ndarray) -> None:
    skip_reason = _should_skip(style)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    engine = ENGINES[style]()
    result = engine.convert(synthetic_photo)

    coverage = ink_coverage(result)
    assert 0.005 <= coverage <= 0.40, (
        f"{style} ink coverage {coverage:.4f} is outside the sanity band "
        "[0.005, 0.40] -- likely a silent total failure (near-blank page "
        "or near-solid ink), not a tuning issue."
    )
