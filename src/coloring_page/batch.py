"""Parallel batch conversion, safe for GPU-resident (VRAM) engines.

A plain ``ProcessPoolExecutor`` over ``convert_image`` would instantiate
one copy of an ML engine's model per worker process -- fine for a small
CPU-only network, but likely to OOM a single GPU running a model like SAM
2. :func:`convert_batch` checks
:attr:`~coloring_page.engines.base.ConversionEngine.requires_serial_execution`
and forces single-process execution for those styles, ignoring any
``jobs`` value the caller requested for them (with a logged warning).

A single file's failure never aborts the batch: every per-file outcome
(success, ``ColoringPageError``, or an unexpected exception) is captured
in a :class:`BatchItemResult` rather than propagated.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from coloring_page.engines.registry import ENGINES
from coloring_page.exceptions import ColoringPageError, EngineUnavailableError
from coloring_page.logging_utils import configure_logging
from coloring_page.profile import Profile

logger = logging.getLogger(__name__)

#: Default parallelism for engines that don't require serial execution,
#: when the caller doesn't request a specific ``jobs`` value. Capped at 4
#: rather than using every core unconditionally, since each worker process
#: also holds a full copy of any classical-engine working buffers.
_DEFAULT_MAX_JOBS = 4


@dataclass(frozen=True)
class BatchItemResult:
    """Outcome of converting one file within a batch.

    Attributes
    ----------
    input_path : Path
    output_path : Path | None
        Where the result was saved, or None if conversion failed.
    error : str | None
        The failure message, or None on success.
    passed_strict : bool | None
        Whether the ``QualityReport`` passed, when strict validation was
        requested; None if strict validation wasn't requested or the
        conversion itself failed.
    """

    input_path: Path
    output_path: Path | None
    error: str | None
    passed_strict: bool | None


@dataclass(frozen=True)
class BatchSummary:
    """Aggregate result of a :func:`convert_batch` call."""

    results: tuple[BatchItemResult, ...]

    @property
    def succeeded(self) -> tuple[BatchItemResult, ...]:
        """Items that converted (and saved) without error."""
        return tuple(r for r in self.results if r.error is None)

    @property
    def failed(self) -> tuple[BatchItemResult, ...]:
        """Items that raised an error during conversion or saving."""
        return tuple(r for r in self.results if r.error is not None)


def _convert_one(
    input_path: Path,
    output_dir: Path,
    profile_json: str,
    fmt: Literal["svg", "pdf", "png"],
    strict: bool,
) -> BatchItemResult:
    """Convert and save one file, capturing (never raising) any failure.

    Runs either in the calling process (serial engines, or ``jobs=1``) or
    in a ``ProcessPoolExecutor`` worker -- ``profile_json`` rather than a
    ``Profile`` instance is what crosses that process boundary, since a
    worker needs to re-seed its own process (seeding does not cross
    process boundaries -- see :func:`coloring_page.seeding.seed_everything`)
    rather than inherit state pickled from the parent.
    """
    from coloring_page.api import convert_image

    profile = Profile.from_json(profile_json)
    if profile.debug_dir is not None:
        # Every file in a batch would otherwise share one DebugSink
        # directory and numbered filename sequence, silently overwriting
        # each other's stages -- namespace by input stem, matching the
        # single-file CLI's own debug output layout.
        profile = profile.with_overrides(debug_dir=profile.debug_dir / input_path.stem)

    try:
        result = convert_image(input_path, profile=profile)
    except (ColoringPageError, FileNotFoundError, OSError) as exc:
        return BatchItemResult(input_path, None, str(exc), None)

    dest = output_dir / f"{input_path.stem}.{fmt}"
    getattr(result, f"save_{fmt}")(dest)
    passed = result.report.passed if strict else None
    return BatchItemResult(input_path, dest, None, passed)


def convert_batch(
    inputs: list[Path],
    output_dir: Path,
    *,
    profile: Profile,
    fmt: Literal["svg", "pdf", "png"] = "png",
    strict: bool = False,
    jobs: int | None = None,
) -> BatchSummary:
    """Convert every file in ``inputs``, in parallel where safe.

    Parameters
    ----------
    inputs : list[Path]
        Input image paths.
    output_dir : Path
        Directory each converted file is saved into, named
        ``<input_stem>.<fmt>``.
    profile : Profile
        Shared conversion configuration for every input.
    fmt : {"svg", "pdf", "png"}, optional
        Output format, by default ``"png"``.
    strict : bool, optional
        If True, each item's ``QualityReport.passed`` is recorded on its
        :class:`BatchItemResult`, by default False.
    jobs : int | None, optional
        Requested worker process count. If None, defaults to
        ``min(4, os.cpu_count())``. Ignored (forced to 1) when
        ``profile.style``'s engine has
        ``requires_serial_execution = True``.

    Returns
    -------
    BatchSummary
    """
    try:
        engine_cls = ENGINES[profile.style]
    except KeyError as exc:
        available = ", ".join(sorted(ENGINES))
        raise EngineUnavailableError(
            f"Unknown style {profile.style!r}. Available styles: {available}"
        ) from exc
    serial = engine_cls.requires_serial_execution

    if serial:
        if jobs not in (None, 1):
            logger.warning(
                "Style %r requires serial execution (a single model instance); ignoring jobs=%d.",
                profile.style,
                jobs,
            )
        effective_jobs = 1
    else:
        effective_jobs = jobs if jobs is not None else min(_DEFAULT_MAX_JOBS, os.cpu_count() or 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_json = profile.to_json()

    if effective_jobs <= 1:
        results = [_convert_one(p, output_dir, profile_json, fmt, strict) for p in inputs]
        return BatchSummary(tuple(results))

    results = []
    with ProcessPoolExecutor(
        max_workers=effective_jobs, initializer=configure_logging, initargs=(False,)
    ) as pool:
        futures = {
            pool.submit(_convert_one, p, output_dir, profile_json, fmt, strict): p for p in inputs
        }
        for future in as_completed(futures):
            input_path = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # a worker crash must not sink the whole batch
                results.append(BatchItemResult(input_path, None, str(exc), None))
    return BatchSummary(tuple(results))
