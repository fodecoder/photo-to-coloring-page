"""Tests for ``Result``'s save_svg/save_pdf/save_png."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coloring_page.drawing import Drawing
from coloring_page.drawing import Path as VectorPath
from coloring_page.page import PageSpec
from coloring_page.result import Result
from coloring_page.validate import validate


@pytest.fixture
def sample_result() -> Result:
    path = VectorPath(
        points=np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32),
        closed=True,
        kind="boundary",
    )
    drawing = Drawing(paths=(path,), aspect_ratio=1.0)
    page = PageSpec(size=(50.0, 50.0), margin_mm=5.0, dpi=100)
    report = validate(drawing, page)
    return Result(drawing=drawing, report=report, timings={"total": 0.1}, page=page)


class TestResultSaveSvg:
    def test_writes_svg_file(self, tmp_path: Path, sample_result: Result) -> None:
        dest = tmp_path / "out.svg"
        sample_result.save_svg(dest)
        assert dest.exists()
        assert dest.read_text(encoding="utf-8").startswith("<?xml")

    def test_creates_parent_dirs(self, tmp_path: Path, sample_result: Result) -> None:
        dest = tmp_path / "nested" / "out.svg"
        sample_result.save_svg(dest)
        assert dest.exists()


class TestResultSavePdf:
    def test_writes_pdf_file(self, tmp_path: Path, sample_result: Result) -> None:
        dest = tmp_path / "out.pdf"
        sample_result.save_pdf(dest)
        assert dest.exists()
        assert dest.read_bytes().startswith(b"%PDF")


class TestResultSavePng:
    def test_writes_png_file(self, tmp_path: Path, sample_result: Result) -> None:
        dest = tmp_path / "out.png"
        sample_result.save_png(dest)
        assert dest.exists()
