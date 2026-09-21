"""Resolve and verify model checkpoint files shared by every ML engine.

Consolidates two things that used to be duplicated per-engine
(``anime2sketch.py``, ``informative_drawings.py``, ``lineart.py`` each had
their own ``_resolve_weights_path`` and their own default cache directory)
and one thing that didn't exist at all: a checksum verified at *load* time,
not only when ``scripts/fetch_weights.py`` first downloads a file. A
checksum checked only at download time says nothing about a file that was
copied in some other way, or that has since been corrupted or modified on
disk -- :func:`verify_checksum` closes that gap by running on every load.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from coloring_page.exceptions import WeightsChecksumError, WeightsMissingError

#: Directory holding every checkpoint's default cache location. Overridable
#: as a whole via this single environment variable -- see .env.example --
#: rather than only per-engine, so a service deploying this package can
#: point every engine at one shared, pre-populated directory (e.g. a
#: read-only volume mount) without setting four separate env vars.
WEIGHTS_DIR_ENV_VAR = "COLORING_PAGE_WEIGHTS_DIR"

DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "coloring_page"

#: SHA256 of every checkpoint this project knows how to fetch, pinned after
#: a verified download -- see ``scripts/fetch_weights.py`` (the original
#: home of this table, now imported from here so download-time and
#: load-time verification share one source of truth) and
#: ``docs/DIAGNOSIS.md``'s Phase 3 for how each entry was obtained. A
#: filename not listed here has no known-good hash to check against:
#: :func:`verify_checksum` treats that as unpinned, not as a failure, since
#: fabricating an expected hash would be worse than not checking one.
CHECKSUMS: dict[str, str] = {
    "sk_model.pth": "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
    "sk_model2.pth": "30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
}


def resolve_weights_dir() -> Path:
    """Return the configured weights cache directory.

    Returns
    -------
    Path
        The directory named by :data:`WEIGHTS_DIR_ENV_VAR`, if set,
        otherwise :data:`DEFAULT_WEIGHTS_DIR`.
    """
    env_dir = os.environ.get(WEIGHTS_DIR_ENV_VAR)
    return Path(env_dir) if env_dir else DEFAULT_WEIGHTS_DIR


def resolve_weights_path(
    filename: str,
    *,
    explicit_path: str | Path | None = None,
    legacy_env_var: str | None = None,
    local_path: Path | None = None,
) -> Path:
    """Pick the checkpoint file to use, in priority order.

    Priority: ``explicit_path`` (a constructor argument) > the engine's own
    deprecated per-file environment variable (``legacy_env_var``, kept for
    cases like Informative Drawings' ``sk_model``/``sk_model2`` A/B that a
    single directory-level variable can't express) >
    ``COLORING_PAGE_WEIGHTS_DIR``/``filename`` > an existing
    ``local_path`` relative to the current working directory > the default
    per-user cache directory. The final fallback is returned even if it
    doesn't exist yet, so callers get a consistent "expected" path to
    report in error messages.

    Parameters
    ----------
    filename : str
        The checkpoint's filename within the weights directory (e.g.
        ``"informative_drawings.pth"``).
    explicit_path : str | Path | None, optional
        A caller-supplied path (e.g. a constructor argument), by default
        None.
    legacy_env_var : str | None, optional
        Name of the engine's own per-file environment variable, checked
        before the shared weights directory, by default None.
    local_path : Path | None, optional
        A ``./weights/...``-style path checked before the default cache
        directory, by default None.

    Returns
    -------
    Path
    """
    if explicit_path is not None:
        return Path(explicit_path)

    if legacy_env_var is not None:
        legacy_value = os.environ.get(legacy_env_var)
        if legacy_value:
            return Path(legacy_value)

    weights_dir_set = os.environ.get(WEIGHTS_DIR_ENV_VAR)
    if weights_dir_set:
        return Path(weights_dir_set) / filename

    if local_path is not None and local_path.exists():
        return local_path

    return DEFAULT_WEIGHTS_DIR / filename


def _sha256(path: Path) -> str:
    """Compute a file's SHA256 hex digest, reading it in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, filename: str) -> None:
    """Verify a checkpoint on disk against its pinned SHA256, failing loudly on mismatch.

    Parameters
    ----------
    path : Path
        Resolved location of the checkpoint file, e.g. from
        :func:`resolve_weights_path`.
    filename : str
        The checkpoint's canonical filename, used to look up its expected
        hash in :data:`CHECKSUMS`.

    Raises
    ------
    WeightsMissingError
        If ``path`` does not exist.
    WeightsChecksumError
        If ``path`` exists but its SHA256 does not match
        ``CHECKSUMS[filename]``.

    Notes
    -----
    A ``filename`` not present in :data:`CHECKSUMS` is treated as
    unpinned: this function returns without checking anything, rather than
    failing or fabricating an expected hash. This is a real gap for
    checkpoints not yet added to that table (see its docstring) -- fail
    the file's existence check via the caller's own ``FileNotFoundError``
    handling if that matters for a given checkpoint.
    """
    expected = CHECKSUMS.get(filename)
    if expected is None:
        return
    if not path.exists():
        raise WeightsMissingError(f"Expected checkpoint not found: {path}")
    digest = _sha256(path)
    if digest != expected:
        raise WeightsChecksumError(
            f"Checksum mismatch for {path} (checkpoint {filename!r}): "
            f"expected {expected}, got {digest}. The file may be corrupted or tampered with."
        )
