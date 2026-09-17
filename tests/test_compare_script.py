"""Smoke test for scripts/compare.py, so it doesn't silently bit-rot."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare import main  # noqa: E402  (import must follow the sys.path tweak above)

from tests.conftest import make_synthetic_photo


def test_compare_writes_contact_sheets_and_metrics_csv(tmp_path: Path) -> None:
    input_dir = tmp_path / "photos"
    output_dir = tmp_path / "compare_out"
    input_dir.mkdir()

    for i in range(2):
        Image.fromarray(make_synthetic_photo()).save(input_dir / f"photo_{i}.png")

    exit_code = main(
        [str(input_dir), str(output_dir), "--styles", "canny,xdog"]
    )

    assert exit_code == 0
    assert (output_dir / "photo_0_compare.png").exists()
    assert (output_dir / "photo_1_compare.png").exists()

    metrics_path = output_dir / "metrics.csv"
    assert metrics_path.exists()
    with metrics_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 4  # 2 images x 2 styles
    assert {row["style"] for row in rows} == {"canny", "xdog"}


def test_compare_empty_directory_returns_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()

    exit_code = main([str(input_dir), str(tmp_path / "out")])

    assert exit_code == 1
