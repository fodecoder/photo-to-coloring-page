"""Tests for weights-directory resolution and load-time checksum verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from coloring_page.exceptions import WeightsChecksumError, WeightsMissingError
from coloring_page.weights import (
    CHECKSUMS,
    DEFAULT_WEIGHTS_DIR,
    resolve_weights_dir,
    resolve_weights_path,
    verify_checksum,
)


class TestResolveWeightsDir:
    def test_defaults_to_per_user_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("COLORING_PAGE_WEIGHTS_DIR", raising=False)
        assert resolve_weights_dir() == DEFAULT_WEIGHTS_DIR

    def test_env_var_overrides_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COLORING_PAGE_WEIGHTS_DIR", str(tmp_path))
        assert resolve_weights_dir() == tmp_path


class TestResolveWeightsPath:
    def test_explicit_path_wins_over_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOME_LEGACY_VAR", str(tmp_path / "legacy.pth"))
        monkeypatch.setenv("COLORING_PAGE_WEIGHTS_DIR", str(tmp_path))
        explicit = tmp_path / "explicit.pth"

        result = resolve_weights_path(
            "model.pth", explicit_path=explicit, legacy_env_var="SOME_LEGACY_VAR"
        )
        assert result == explicit

    def test_legacy_env_var_wins_over_shared_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        legacy_path = tmp_path / "legacy.pth"
        monkeypatch.setenv("SOME_LEGACY_VAR", str(legacy_path))
        monkeypatch.setenv("COLORING_PAGE_WEIGHTS_DIR", str(tmp_path))

        result = resolve_weights_path("model.pth", legacy_env_var="SOME_LEGACY_VAR")
        assert result == legacy_path

    def test_shared_weights_dir_wins_over_local_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SOME_LEGACY_VAR", raising=False)
        monkeypatch.setenv("COLORING_PAGE_WEIGHTS_DIR", str(tmp_path))
        local_path = tmp_path / "local" / "model.pth"
        local_path.parent.mkdir()
        local_path.write_bytes(b"x")

        result = resolve_weights_path(
            "model.pth", legacy_env_var="SOME_LEGACY_VAR", local_path=local_path
        )
        assert result == tmp_path / "model.pth"

    def test_local_path_wins_over_default_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SOME_LEGACY_VAR", raising=False)
        monkeypatch.delenv("COLORING_PAGE_WEIGHTS_DIR", raising=False)
        local_path = tmp_path / "local" / "model.pth"
        local_path.parent.mkdir()
        local_path.write_bytes(b"x")

        result = resolve_weights_path(
            "model.pth", legacy_env_var="SOME_LEGACY_VAR", local_path=local_path
        )
        assert result == local_path

    def test_falls_back_to_default_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_LEGACY_VAR", raising=False)
        monkeypatch.delenv("COLORING_PAGE_WEIGHTS_DIR", raising=False)

        result = resolve_weights_path(
            "model.pth", legacy_env_var="SOME_LEGACY_VAR", local_path=Path("does-not-exist.pth")
        )
        assert result == DEFAULT_WEIGHTS_DIR / "model.pth"


class TestVerifyChecksum:
    def test_missing_file_raises_weights_missing(self, tmp_path: Path) -> None:
        pinned_name = next(iter(CHECKSUMS))
        with pytest.raises(WeightsMissingError):
            verify_checksum(tmp_path / "does-not-exist.pth", pinned_name)

    def test_mismatched_checksum_raises(self, tmp_path: Path) -> None:
        pinned_name = next(iter(CHECKSUMS))
        path = tmp_path / pinned_name
        path.write_bytes(b"not the real checkpoint")

        with pytest.raises(WeightsChecksumError):
            verify_checksum(path, pinned_name)

    def test_unpinned_filename_is_a_noop(self, tmp_path: Path) -> None:
        # A filename with no entry in CHECKSUMS has nothing to check
        # against -- this must not raise, even though the file doesn't
        # exist on disk.
        verify_checksum(tmp_path / "does-not-exist.pth", "totally-unknown-checkpoint.pth")
