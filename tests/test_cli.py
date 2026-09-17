"""Tests for the command-line interface, including batch mode."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from coloring_page.cli import main
from tests.conftest import make_synthetic_photo


def test_single_file_conversion(tmp_path: Path, synthetic_photo_path: Path) -> None:
    output_path = tmp_path / "out.png"

    exit_code = main([str(synthetic_photo_path), str(output_path), "--style", "canny"])

    assert exit_code == 0
    assert output_path.exists()


def test_invalid_input_path_returns_error(tmp_path: Path) -> None:
    exit_code = main([str(tmp_path / "missing.png"), str(tmp_path / "out.png")])
    assert exit_code == 1


def test_unsupported_format_returns_error(tmp_path: Path) -> None:
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not an image")

    exit_code = main([str(bogus), str(tmp_path / "out.png")])
    assert exit_code == 1


def test_batch_mode_processes_multiple_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    for i in range(3):
        Image.fromarray(make_synthetic_photo()).save(input_dir / f"photo_{i}.png")

    exit_code = main([str(input_dir), str(output_dir), "--style", "adaptive"])

    assert exit_code == 0
    outputs = sorted(output_dir.glob("*_coloring.png"))
    assert len(outputs) == 3


def test_batch_mode_empty_directory_returns_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()

    exit_code = main([str(input_dir), str(tmp_path / "out")])
    assert exit_code == 1
