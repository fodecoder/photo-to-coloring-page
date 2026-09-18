"""Tests for the command-line interface, including batch mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from coloring_page import cli
from coloring_page.cli import build_parser, main
from coloring_page.pipeline import DEFAULT_WORKING_DIMENSION, default_line_thickness
from tests.conftest import make_synthetic_photo


def test_max_dimension_defaults_to_working_resolution() -> None:
    args = build_parser().parse_args(["in.png", "out.png"])
    assert args.max_dimension == DEFAULT_WORKING_DIMENSION


def test_thickness_defaults_to_none_before_resolution() -> None:
    # The CLI flag itself stays unset (None) until run() resolves it from
    # --max-dimension, since the right default depends on that value.
    args = build_parser().parse_args(["in.png", "out.png"])
    assert args.thickness is None


def test_thickness_defaults_to_working_resolution_derived_value(
    tmp_path: Path, synthetic_photo_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    original_convert_image = cli.convert_image

    def spy_convert_image(*args: Any, **kwargs: Any) -> Any:
        captured["line_thickness"] = kwargs["line_thickness"]
        captured["max_dimension"] = kwargs["max_dimension"]
        return original_convert_image(*args, **kwargs)

    monkeypatch.setattr(cli, "convert_image", spy_convert_image)

    exit_code = main([str(synthetic_photo_path), str(tmp_path / "out.png")])

    assert exit_code == 0
    assert captured["line_thickness"] == default_line_thickness(captured["max_dimension"])


def test_thickness_explicit_value_overrides_default(
    tmp_path: Path, synthetic_photo_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    original_convert_image = cli.convert_image

    def spy_convert_image(*args: Any, **kwargs: Any) -> Any:
        captured["line_thickness"] = kwargs["line_thickness"]
        return original_convert_image(*args, **kwargs)

    monkeypatch.setattr(cli, "convert_image", spy_convert_image)

    exit_code = main([str(synthetic_photo_path), str(tmp_path / "out.png"), "--thickness", "7"])

    assert exit_code == 0
    assert captured["line_thickness"] == 7


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


def test_batch_mode_writes_png_regardless_of_input_extension(tmp_path: Path) -> None:
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    Image.fromarray(make_synthetic_photo()).save(input_dir / "photo.jpg")

    exit_code = main([str(input_dir), str(output_dir), "--style", "canny"])

    assert exit_code == 0
    assert (output_dir / "photo_coloring.png").exists()
    assert not (output_dir / "photo_coloring.jpg").exists()


def test_single_file_output_respects_explicit_extension(
    tmp_path: Path, synthetic_photo_path: Path
) -> None:
    output_path = tmp_path / "out.jpg"

    exit_code = main([str(synthetic_photo_path), str(output_path), "--style", "canny"])

    assert exit_code == 0
    assert output_path.exists()


def test_batch_mode_empty_directory_returns_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()

    exit_code = main([str(input_dir), str(tmp_path / "out")])
    assert exit_code == 1


def test_debug_dir_creates_stage_output_files(tmp_path: Path, synthetic_photo_path: Path) -> None:
    debug_dir = tmp_path / "debug"

    exit_code = main(
        [
            str(synthetic_photo_path),
            str(tmp_path / "out.png"),
            "--style",
            "canny",
            "--debug-dir",
            str(debug_dir),
        ]
    )

    assert exit_code == 0
    assert any(debug_dir.iterdir())


def test_debug_dir_namespaces_per_file_in_batch_mode(tmp_path: Path) -> None:
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "out"
    debug_dir = tmp_path / "debug"
    input_dir.mkdir()

    for i in range(2):
        Image.fromarray(make_synthetic_photo()).save(input_dir / f"photo_{i}.png")

    exit_code = main(
        [
            str(input_dir),
            str(output_dir),
            "--style",
            "canny",
            "--debug-dir",
            str(debug_dir),
        ]
    )

    assert exit_code == 0
    assert (debug_dir / "photo_0").exists()
    assert (debug_dir / "photo_1").exists()
    assert any((debug_dir / "photo_0").iterdir())
