"""Tests for ``Profile`` (de)serialization and the detail-level filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coloring_page.drawing import Drawing
from coloring_page.drawing import Path as VectorPath
from coloring_page.page import PageSpec
from coloring_page.profile import DETAIL_PRESETS, Profile, apply_detail


def _long_open_path(length_fraction: float = 0.5) -> VectorPath:
    return VectorPath(
        points=np.array([[0.0, 0.0], [length_fraction, 0.0]], dtype=np.float32),
        closed=False,
        kind="boundary",
    )


def _medium_open_path() -> VectorPath:
    # Length 0.03 normalized units: on a 100x100mm page (scale=100mm/unit)
    # that's 3mm -- above adult's 1.5mm floor, below toddler's 6mm floor,
    # so the two presets are expected to disagree on this path.
    return VectorPath(
        points=np.array([[0.0, 0.0], [0.03, 0.0]], dtype=np.float32),
        closed=False,
        kind="boundary",
    )


class TestProfileJsonRoundTrip:
    def test_default_profile_round_trips(self) -> None:
        profile = Profile()
        restored = Profile.from_json(profile.to_json())
        assert restored == profile

    def test_named_page_size_round_trips(self) -> None:
        profile = Profile(page=PageSpec(size="A5", orientation="landscape"))
        restored = Profile.from_json(profile.to_json())
        assert restored.page.size == "A5"
        assert restored.page.orientation == "landscape"

    def test_explicit_tuple_page_size_round_trips(self) -> None:
        profile = Profile(page=PageSpec(size=(100.0, 150.0)))
        restored = Profile.from_json(profile.to_json())
        assert restored.page.size == (100.0, 150.0)

    def test_debug_dir_none_round_trips(self) -> None:
        profile = Profile(debug_dir=None)
        restored = Profile.from_json(profile.to_json())
        assert restored.debug_dir is None

    def test_debug_dir_set_round_trips(self) -> None:
        profile = Profile(debug_dir=Path("some/debug/dir"))
        restored = Profile.from_json(profile.to_json())
        assert restored.debug_dir == Path("some/debug/dir")

    def test_seed_none_round_trips(self) -> None:
        profile = Profile(seed=None)
        restored = Profile.from_json(profile.to_json())
        assert restored.seed is None


class TestProfileFromJsonRejectsUnknownDetail:
    def test_unknown_detail_level_raises(self) -> None:
        with pytest.raises(ValueError, match="detail"):
            Profile.from_json('{"detail": "grownup"}')


class TestProfileWithOverrides:
    def test_overrides_only_named_fields(self) -> None:
        base = Profile(style="chained", detail="child")
        changed = base.with_overrides(style="canny")
        assert changed.style == "canny"
        assert changed.detail == "child"


class TestApplyDetail:
    def test_toddler_drops_more_than_adult(self) -> None:
        drawing = Drawing(paths=(_long_open_path(), _medium_open_path()), aspect_ratio=1.0)
        page = PageSpec(size=(100.0, 100.0), margin_mm=0.0)
        toddler_profile = Profile(detail="toddler", page=page)
        adult_profile = Profile(detail="adult", page=page)

        toddler_result = apply_detail(drawing, toddler_profile)
        adult_result = apply_detail(drawing, adult_profile)

        assert len(toddler_result.paths) < len(adult_result.paths)

    def test_source_area_none_is_not_filtered_by_area(self) -> None:
        # A path with no source_area must survive the area filter
        # regardless of preset, since the filter is a documented no-op
        # for engines with no region concept.
        drawing = Drawing(paths=(_long_open_path(),), aspect_ratio=1.0)
        profile = Profile(detail="toddler", page=PageSpec(size=(100.0, 100.0), margin_mm=0.0))

        result = apply_detail(drawing, profile)

        assert len(result.paths) == 1

    def test_every_detail_level_has_presets(self) -> None:
        assert set(DETAIL_PRESETS) == {"toddler", "child", "adult"}
