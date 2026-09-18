"""Smoke test for scripts/compare.py, so it doesn't silently bit-rot."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare import (  # noqa: E402  (import must follow the sys.path tweak above)
    REFERENCE_PAIRS,
    main,
)

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
    # No reference pair known for these filenames, so the ref_* columns
    # are present but left blank rather than omitted.
    assert all(row["ref_f1"] == "" for row in rows)


def test_compare_empty_directory_returns_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()

    exit_code = main([str(input_dir), str(tmp_path / "out")])

    assert exit_code == 1


def test_compare_scores_known_reference_pair(tmp_path: Path) -> None:
    input_dir = tmp_path / "photos"
    ref_dir = tmp_path / "refs"
    output_dir = tmp_path / "compare_out"
    input_dir.mkdir()
    ref_dir.mkdir()

    input_name, ref_name = next(iter(REFERENCE_PAIRS.items()))
    photo = make_synthetic_photo()
    Image.fromarray(photo).save(input_dir / input_name)
    # A trivial reference: any single-channel image works for boundary
    # scoring, it doesn't need to be a "real" line drawing for this test.
    Image.fromarray(photo).convert("L").save(ref_dir / ref_name)

    exit_code = main(
        [str(input_dir), str(output_dir), "--styles", "canny", "--ref-dir", str(ref_dir)]
    )

    assert exit_code == 0
    with (output_dir / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["ref_f1"] != ""
    assert 0.0 <= float(rows[0]["ref_f1"]) <= 1.0
