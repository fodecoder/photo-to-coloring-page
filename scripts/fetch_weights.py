"""Download Informative Drawings pretrained weights for automation/CI use.

The upstream project (https://github.com/carolineec/informative-drawings)
only distributes weights via a Google Drive link, which cannot be scripted
without manual interaction. The same weights are redistributed on the
Hugging Face Hub as `lllyasviel/Annotators`, used by ControlNet's "lineart"
preprocessor and downloadable without a login. This is a deliberate,
documented departure from that project's originally-preferred official
source -- see THIRD_PARTY_LICENSES.md for why, and for the residual
licensing risk this accepts.

Two Informative Drawings checkpoints are available: `sk_model.pth` (fine
detail) and `sk_model2.pth` (coarse detail). Which one performs best on
this project's reference images is an empirical question -- see the README
for how to measure both with `scripts/compare.py` and choose.

A third checkpoint, `sam2.1_hiera_small.pt`, is SAM 2.1's Hiera-small
checkpoint, redistributed on its own official Hugging Face repository
(unlike Informative Drawings, no mirror is needed). It's used by
`scripts/spike_segmentation.py`'s region-partition spike -- see
docs/DIAGNOSIS.md -- and by the `lineart` engine's Stage A
(src/coloring_page/engines/lineart.py).

A fourth checkpoint, `netG.pth`, is controlnet_aux's `lineart_anime`
preprocessor weights, used by the `lineart` engine's Stage B for fine
internal detail (facial features, folds, deliberate patterns) SAM's
region partition alone can't see -- see docs/PRODUCTION-PROMPTS.md's
Prompt 3. Kept as `netG.pth` (not renamed like the Informative Drawings
checkpoints) since controlnet_aux's own loader expects to find it under
that exact name.

Usage
-----
::

    python scripts/fetch_weights.py --filename sk_model2.pth
    python scripts/fetch_weights.py --filename sk_model.pth --dest weights/informative_drawings.pth
    python scripts/fetch_weights.py --filename sam2.1_hiera_small.pt
    python scripts/fetch_weights.py --filename netG.pth --dest weights/netG.pth
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from coloring_page.weights import CHECKSUMS, _sha256, resolve_weights_dir

#: Per-checkpoint source repository and default destination filename. See
#: module docstring and THIRD_PARTY_LICENSES.md for the license caveat
#: attached to the lllyasviel/Annotators mirror.
_CHECKPOINTS: dict[str, dict[str, str]] = {
    "sk_model.pth": {
        "repo_id": "lllyasviel/Annotators",
        "dest_name": "informative_drawings.pth",
    },
    "sk_model2.pth": {
        "repo_id": "lllyasviel/Annotators",
        "dest_name": "informative_drawings.pth",
    },
    "sam2.1_hiera_small.pt": {
        "repo_id": "facebook/sam2.1-hiera-small",
        "dest_name": "sam2.1_hiera_small.pt",
    },
    # controlnet_aux's LineartAnimeDetector loads this same filename from
    # this same repository by default (its own from_pretrained call), and
    # expects to find it under exactly this name in whatever directory is
    # handed to from_pretrained() -- kept as "netG.pth" here too (rather
    # than renamed like the Informative Drawings checkpoints) so the
    # `lineart` engine's weights directory just works when pointed at
    # LineartAnimeDetector.from_pretrained(), while still going through
    # this project's own weights-path convention (explicit path / env var
    # / weights/ / per-user cache, see lineart.py's _resolve_weights_path)
    # instead of relying on controlnet_aux's separate download/cache.
    "netG.pth": {
        "repo_id": "lllyasviel/Annotators",
        "dest_name": "netG.pth",
    },
}

#: Cache directory every checkpoint's default destination lives under,
#: matching each engine's own fallback resolution path (see
#: coloring_page.weights.resolve_weights_dir, which this respects the
#: COLORING_PAGE_WEIGHTS_DIR override of) so a fetched file is found
#: automatically without also setting a per-engine env var.
_CACHE_DIR = resolve_weights_dir()

#: SHA256 of each checkpoint this project knows how to fetch -- imported
#: from coloring_page.weights so this script's download-time check and
#: every engine's load-time check (coloring_page.weights.verify_checksum)
#: share one pinned source of truth. See that module's docstring and
#: docs/DIAGNOSIS.md's Phase 3 for how each entry was obtained:
#: sk_model.pth ("fine" detail) measured a higher f1_normalized on 2 of the
#: 3 reference pairs and is this project's chosen Informative Drawings
#: default; sk_model2.pth ("coarse") is kept fetchable and pinned too, for
#: anyone who wants to compare again on their own images. A filename not in
#: this dict has no expected hash to check against -- fetch_weights() falls
#: back to printing it instead, never fabricating one.
_PINNED_SHA256 = CHECKSUMS

#: Fallback base path used only to derive suffixed destinations when
#: fetching multiple checkpoints that share the same default ``dest_name``
#: (the two Informative Drawings checkpoints) without an explicit --dest.
_DEFAULT_DEST = _CACHE_DIR / "informative_drawings.pth"


def fetch_weights(
    filename: str, *, dest: Path | None = None, expected_sha256: str | None = None
) -> Path:
    """Download one checkpoint from its Hugging Face repository and copy it to ``dest``.

    Parameters
    ----------
    filename : str
        Checkpoint filename, one of the keys of ``_CHECKPOINTS`` (e.g.
        ``"sk_model.pth"``, ``"sk_model2.pth"``, or
        ``"sam2.1_hiera_small.pt"``); its source repository is looked up
        from that registry.
    dest : Path | None, optional
        Where to copy the downloaded file. If omitted (the default), uses
        ``~/.cache/coloring_page/<dest_name>``, the same path each
        engine/script falls back to when no weights path is given
        explicitly (e.g. ``InformativeDrawingsEngine``).
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

    if filename not in _CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint {filename!r}; expected one of {sorted(_CHECKPOINTS)}")
    checkpoint = _CHECKPOINTS[filename]
    if dest is None:
        dest = _CACHE_DIR / checkpoint["dest_name"]

    downloaded_path = Path(hf_hub_download(repo_id=checkpoint["repo_id"], filename=filename))
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
        description="Download pretrained checkpoints used by this project's optional engines."
    )
    parser.add_argument(
        "--filename",
        choices=sorted(_CHECKPOINTS),
        default=None,
        help="Which checkpoint to fetch (default: both Informative Drawings checkpoints, "
        "saved as -sk_model and -sk_model2 suffixed copies next to --dest so both can be "
        "compared; does not include sam2.1_hiera_small.pt, which must be requested explicitly).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Where to copy the downloaded file (default: "
        f"{_CACHE_DIR}/<checkpoint's own filename>, e.g. {_DEFAULT_DEST} for the Informative "
        "Drawings checkpoints).",
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
                base = args.dest if args.dest is not None else _DEFAULT_DEST
                dest = base.with_name(f"{base.stem}-{filename.removesuffix('.pth')}.pth")
            else:
                dest = args.dest
            saved_to = fetch_weights(filename, dest=dest, expected_sha256=args.expected_sha256)
            print(f"Saved {filename} -> {saved_to}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
