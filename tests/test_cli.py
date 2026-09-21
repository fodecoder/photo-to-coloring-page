"""Tests for the command-line interface, including batch mode."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from coloring_page import api as api_module
from coloring_page import cli
from coloring_page.cli import build_parser, main
from coloring_page.exceptions import ImageTooLargeError, QualityGateError, UnsupportedImageError
from coloring_page.profile import Profile
from coloring_page.validate import QualityReport
from tests.conftest import make_synthetic_photo


def test_experimental_styles_hidden_from_style_metavar_by_default() -> None:
    parser = build_parser([])
    style_action = next(a for a in parser._actions if a.dest == "style")
    assert "canny" not in (style_action.metavar or "")


def test_show_experimental_flag_reveals_experimental_styles_in_metavar() -> None:
    parser = build_parser(["--show-experimental"])
    style_action = next(a for a in parser._actions if a.dest == "style")
    assert "canny" in (style_action.metavar or "")


def test_experimental_style_remains_selectable_without_show_experimental(
    tmp_path: Path, synthetic_photo_path: Path
) -> None:
    # Hidden from --help's displayed list, but never rejected as an
    # invalid --style choice -- see build_parser's docstring/comments.
    exit_code = main([str(synthetic_photo_path), str(tmp_path / "out.png"), "--style", "canny"])
    assert exit_code == 0


def test_single_file_conversion(tmp_path: Path, synthetic_photo_path: Path) -> None:
    output_path = tmp_path / "out.png"

    exit_code = main([str(synthetic_photo_path), str(output_path), "--style", "canny"])

    assert exit_code == 0
    assert output_path.exists()


def test_invalid_input_path_returns_error(tmp_path: Path) -> None:
    exit_code = main([str(tmp_path / "missing.png"), str(tmp_path / "out.png")])
    assert exit_code == 1


def test_unsupported_format_returns_dedicated_exit_code(tmp_path: Path) -> None:
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not an image")

    exit_code = main([str(bogus), str(tmp_path / "out.png")])
    assert exit_code == cli._EXIT_CODES[UnsupportedImageError]


def test_image_too_large_returns_dedicated_exit_code(
    tmp_path: Path, synthetic_photo_path: Path
) -> None:
    exit_code = main([str(synthetic_photo_path), str(tmp_path / "out.png"), "--max-pixels", "1"])
    assert exit_code == cli._EXIT_CODES[ImageTooLargeError]


def test_batch_mode_processes_multiple_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    for i in range(3):
        Image.fromarray(make_synthetic_photo()).save(input_dir / f"photo_{i}.png")

    exit_code = main([str(input_dir), str(output_dir), "--style", "adaptive"])

    assert exit_code == 0
    outputs = sorted(output_dir.glob("*.png"))
    assert len(outputs) == 3


def test_batch_mode_writes_configured_format_regardless_of_input_extension(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    Image.fromarray(make_synthetic_photo()).save(input_dir / "photo.jpg")

    exit_code = main([str(input_dir), str(output_dir), "--style", "canny", "--format", "svg"])

    assert exit_code == 0
    assert (output_dir / "photo.svg").exists()


def test_single_file_output_respects_explicit_extension(
    tmp_path: Path, synthetic_photo_path: Path
) -> None:
    output_path = tmp_path / "out.svg"

    exit_code = main([str(synthetic_photo_path), str(output_path), "--style", "canny"])

    assert exit_code == 0
    assert output_path.exists()


def test_format_flag_overrides_output_extension(tmp_path: Path, synthetic_photo_path: Path) -> None:
    output_path = tmp_path / "out.png"

    exit_code = main(
        [str(synthetic_photo_path), str(output_path), "--style", "canny", "--format", "pdf"]
    )

    assert exit_code == 0
    # --format wins, so the actual bytes written are a PDF despite the .png name.
    assert output_path.read_bytes().startswith(b"%PDF")


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
    saved = {p.name for p in debug_dir.iterdir()}
    assert any("drawing_pre_filter" in name for name in saved)
    assert any("drawing_post_filter" in name for name in saved)


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
            "--jobs",
            "1",
        ]
    )

    assert exit_code == 0
    assert (debug_dir / "photo_0").exists()
    assert (debug_dir / "photo_1").exists()
    assert any((debug_dir / "photo_0").iterdir())


def _fake_report(*, passed: bool) -> QualityReport:
    return QualityReport(
        ink_coverage=0.05,
        enclosed_regions=1,
        leaking_regions=0 if passed else 1,
        min_region_area_mm2=10.0,
        region_area_mm2_p5=10.0,
        dangling_endpoints=0,
    )


def test_strict_flag_defaults_to_false() -> None:
    args = build_parser().parse_args(["in.png", "out.png"])
    assert args.strict is False


def test_strict_flag_parses_true() -> None:
    args = build_parser().parse_args(["in.png", "out.png", "--strict"])
    assert args.strict is True


def test_strict_failure_returns_dedicated_exit_code_for_single_file(
    tmp_path: Path, synthetic_photo_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "validate", lambda drawing, spec: _fake_report(passed=False))

    exit_code = main([str(synthetic_photo_path), str(tmp_path / "out.png"), "--strict"])

    assert exit_code == cli._EXIT_CODES[QualityGateError]


def test_strict_success_returns_ok_for_single_file(
    tmp_path: Path, synthetic_photo_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "validate", lambda drawing, spec: _fake_report(passed=True))

    exit_code = main([str(synthetic_photo_path), str(tmp_path / "out.png"), "--strict"])

    assert exit_code == 0


def test_without_strict_flag_output_still_saved(tmp_path: Path, synthetic_photo_path: Path) -> None:
    output_path = tmp_path / "out.png"
    exit_code = main([str(synthetic_photo_path), str(output_path)])

    assert exit_code == 0
    assert output_path.exists()


def test_strict_failure_returns_dedicated_exit_code_for_batch_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    Image.fromarray(make_synthetic_photo()).save(input_dir / "photo.png")

    monkeypatch.setattr(api_module, "validate", lambda drawing, spec: _fake_report(passed=False))

    exit_code = main([str(input_dir), str(output_dir), "--strict", "--jobs", "1"])

    assert exit_code == cli._EXIT_CODES[QualityGateError]
    # The failing file is still converted and written -- --strict is a
    # gate on the exit code, not on whether output gets produced.
    assert (output_dir / "photo.png").exists()


def test_strict_success_returns_ok_for_batch_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    Image.fromarray(make_synthetic_photo()).save(input_dir / "photo.png")

    monkeypatch.setattr(api_module, "validate", lambda drawing, spec: _fake_report(passed=True))

    exit_code = main([str(input_dir), str(output_dir), "--strict", "--jobs", "1"])

    assert exit_code == 0


def test_deprecated_thickness_flag_is_accepted_and_warns(
    tmp_path: Path, synthetic_photo_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = main([str(synthetic_photo_path), str(tmp_path / "out.png"), "--thickness", "7"])

    assert exit_code == 0
    assert any("deprecated" in record.message for record in caplog.records)


def test_profile_flag_loads_base_configuration(tmp_path: Path, synthetic_photo_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(Profile(style="canny", detail="toddler").to_json())

    args = build_parser().parse_args(
        [str(synthetic_photo_path), str(tmp_path / "out.png"), "--profile", str(profile_path)]
    )
    profile = cli._build_profile(args)

    assert profile.style == "canny"
    assert profile.detail == "toddler"


def test_explicit_flag_overrides_profile_file(tmp_path: Path, synthetic_photo_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(Profile(style="canny").to_json())

    args = build_parser().parse_args(
        [
            str(synthetic_photo_path),
            str(tmp_path / "out.png"),
            "--profile",
            str(profile_path),
            "--style",
            "adaptive",
        ]
    )
    profile = cli._build_profile(args)

    assert profile.style == "adaptive"


def test_jobs_flag_is_passed_through_to_batch(tmp_path: Path) -> None:
    args = build_parser().parse_args(["in", "out", "--jobs", "2"])
    assert args.jobs == 2


def test_seed_flag_defaults_to_none_before_resolution() -> None:
    args = build_parser().parse_args(["in.png", "out.png"])
    assert args.seed is None
