"""Download Informative Drawings pretrained weights for automation/CI use.

The upstream project (https://github.com/carolineec/informative-drawings)
only distributes weights via a Google Drive link, which cannot be scripted
without manual interaction. The same weights are redistributed on the
Hugging Face Hub as `lllyasviel/Annotators`, used by ControlNet's "lineart"
preprocessor and downloadable without a login. This is a deliberate,
documented departure from that project's originally-preferred official
source -- see THIRD_PARTY_LICENSES.md for why, and for the residual
licensing risk this accepts.

Two checkpoints are available: `sk_model.pth` (fine detail) and
`sk_model2.pth` (coarse detail). Which one performs best on this project's
reference images is an empirical question -- see the README for how to
measure both with `scripts/compare.py` and choose.

Usage
-----
::

    python scripts/fetch_weights.py --filename sk_model2.pth
    python scripts/fetch_weights.py --filename sk_model.pth --dest weights/informative_drawings.pth
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

#: Repository the weights are redistributed from. See module docstring and
#: THIRD_PARTY_LICENSES.md for the license caveat attached to this source.
_REPO_ID = "lllyasviel/Annotators"

#: Default destination, matching informative_drawings.py's own fallback
#: resolution path (DEFAULT_WEIGHTS_PATH) so a fetched file is found
#: automatically without also setting the env var.
_DEFAULT_DEST = Path.home() / ".cache" / "coloring_page" / "informative_drawings.pth"

#: SHA256 of each checkpoint on the lllyasviel/Annotators mirror, pinned
#: after comparing both with scripts/compare.py (see docs/DIAGNOSIS.md's
#: Phase 3 and the commit that added this): sk_model.pth ("fine" detail)
#: measured a higher f1_normalized on 2 of the 3 reference pairs and is
#: this project's chosen default; sk_model2.pth ("coarse") is kept
#: fetchable and pinned too, for anyone who wants to compare again on
#: their own images. A filename not in this dict has no expected hash to
#: check against -- fetch_weights() falls back to printing it instead.
_PINNED_SHA256 = {
    "sk_model.pth": "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
    "sk_model2.pth": "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
}


def _sha256(path: Path) -> str:
    """Compute a file's SHA256 hex digest, reading it in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weights(
    filename: str, *, dest: Path = _DEFAULT_DEST, expected_sha256: str | None = None
) -> Path:
    """Download one checkpoint from the Hugging Face mirror and copy it to ``dest``.

    Parameters
    ----------
    filename : str
        Checkpoint filename in the ``lllyasviel/Annotators`` repository,
        e.g. ``"sk_model.pth"`` or ``"sk_model2.pth"``.
    dest : Path, optional
        Where to copy the downloaded file, by default
        ``~/.cache/coloring_page/informative_drawings.pth`` (the same path
        ``InformativeDrawingsEngine`` falls back to when no weights path is
        given explicitly).
    expected_sha256 : str | None, optional
        If given, the download is verified against this hash and a
        ``ValueError`` is raised on mismatch. If omitted (the default),
        falls back to ``_PINNED_SHA256[filename]`` for a known checkpoint;
        for an unrecognized filename, no value is assumed or fabricated --
        the computed hash is printed instead, so it can be pinned once
        confirmed.

    Returns
    -------
    Path
        ``dest``, once the file has been copied there.
    """
    from huggingface_hub import hf_hub_download

    downloaded_path = Path(hf_hub_download(repo_id=_REPO_ID, filename=filename))
    digest = _sha256(downloaded_path)

    if expected_sha256 is None:
        expected_sha256 = _PINNED_SHA256.get(filename)

    if expected_sha256 is not None:
        if digest != expected_sha256:
            raise ValueError(
                f"SHA256 mismatch for {filename}: expected {expected_sha256}, got {digest}"
            )
    else:
        print(f"{filename}: no expected hash provided -- pin this value once confirmed: {digest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded_path, dest)
    return dest


def build_parser() -> argparse.ArgumentParser:
    """Construct this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download Informative Drawings pretrained weights from the "
            "lllyasviel/Annotators Hugging Face mirror."
        )
    )
    parser.add_argument(
        "--filename",
        choices=["sk_model.pth", "sk_model2.pth"],
        default=None,
        help="Which checkpoint to fetch (default: both, saved as -sk_model and -sk_model2 "
        "suffixed copies next to --dest so both can be compared).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=_DEFAULT_DEST,
        help=f"Where to copy the downloaded file (default: {_DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Verify the download against this SHA256 hash (default: none -- printed instead).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Script entry point.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to parse instead of ``sys.argv[1:]``, by default None.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` on a reported failure.
    """
    args = build_parser().parse_args(argv)
    filenames = [args.filename] if args.filename is not None else ["sk_model.pth", "sk_model2.pth"]

    if len(filenames) > 1 and args.expected_sha256 is not None:
        print("Error: --expected-sha256 requires a single --filename", file=sys.stderr)
        return 1

    try:
        for filename in filenames:
            if len(filenames) > 1:
                dest = args.dest.with_name(f"{args.dest.stem}-{filename.removesuffix('.pth')}.pth")
            else:
                dest = args.dest
            fetch_weights(filename, dest=dest, expected_sha256=args.expected_sha256)
            print(f"Saved {filename} -> {dest}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
