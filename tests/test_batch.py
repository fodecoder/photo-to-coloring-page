"""Tests for parallel batch conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from coloring_page.batch import convert_batch
from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.engines.registry import ENGINES
from coloring_page.profile import Profile
from tests.conftest import make_synthetic_photo


class _FakeSerialEngine(ConversionEngine):
    """Duck-types a GPU-resident engine, for testing the serial-forcing path in isolation."""

    name = "fake-serial"
    requires_serial_execution = True

    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> object:
        from coloring_page.drawing import Drawing
        from coloring_page.drawing import Path as VectorPath

        path = VectorPath(
            points=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            closed=False,
            kind="boundary",
        )
        return Drawing(paths=(path,), aspect_ratio=image.shape[1] / image.shape[0])


def _write_photos(input_dir: Path, count: int) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = input_dir / f"photo_{i}.png"
        Image.fromarray(make_synthetic_photo()).save(path)
        paths.append(path)
    return paths


class TestConvertBatchContinuesOnFailure:
    def test_one_bad_file_does_not_block_the_rest(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "photos"
        output_dir = tmp_path / "out"
        good_paths = _write_photos(input_dir, 2)
        bad_path = input_dir / "not_an_image.png"
        bad_path.write_text("not an image")

        summary = convert_batch(
            [*good_paths, bad_path], output_dir, profile=Profile(style="canny"), jobs=1
        )

        assert len(summary.succeeded) == 2
        assert len(summary.failed) == 1
        assert summary.failed[0].input_path == bad_path


class TestConvertBatchForcesSerialForGpuEngines:
    def test_serial_engine_ignores_requested_jobs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(ENGINES, "fake-serial", _FakeSerialEngine)

        input_dir = tmp_path / "photos"
        output_dir = tmp_path / "out"
        paths = _write_photos(input_dir, 2)

        summary = convert_batch(paths, output_dir, profile=Profile(style="fake-serial"), jobs=4)

        assert len(summary.succeeded) == 2


class TestConvertBatchRespectsJobs:
    def test_classical_engine_runs_with_requested_jobs(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "photos"
        output_dir = tmp_path / "out"
        paths = _write_photos(input_dir, 2)

        summary = convert_batch(paths, output_dir, profile=Profile(style="canny"), jobs=1)

        assert len(summary.succeeded) == 2
        assert all(p.output_path is not None and p.output_path.exists() for p in summary.succeeded)
