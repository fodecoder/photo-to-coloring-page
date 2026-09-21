"""Structured, per-stage logging shared by the public API and the CLI.

Before this module, the only user-facing output anywhere in the package
was ``print()`` calls confined to ``cli.py`` -- there was no way for a
caller embedding :func:`coloring_page.convert_image` in a service to get
diagnostic output at all, structured or not. ``configure_logging`` is
called once, by the CLI; :func:`stage_timer` is used by
:mod:`coloring_page.api` (and may be used by any engine) to log a
stage's start/duration uniformly.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

#: Root logger name for this package; every module-level logger below it
#: (``coloring_page.api``, ``coloring_page.engines.lineart``, ...) inherits
#: whatever level/handler :func:`configure_logging` sets here.
LOGGER_NAME = "coloring_page"


def configure_logging(verbose: bool) -> None:
    """Configure this package's root logger to write to stderr.

    Parameters
    ----------
    verbose : bool
        If True, emit DEBUG-level messages (including per-stage "start"
        events); otherwise INFO-level only (stage completions and
        warnings/errors).
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@contextmanager
def stage_timer(logger: logging.Logger, stage: str, timings: dict[str, float]) -> Iterator[None]:
    """Time one pipeline stage, logging its start and completion.

    Parameters
    ----------
    logger : logging.Logger
        Logger to write the start/completion messages to.
    stage : str
        Short, stable name for this stage (e.g. ``"resize"``), used both
        in the log message and as ``timings``' key.
    timings : dict[str, float]
        Mutated in place: ``timings[stage]`` is set to the elapsed time in
        seconds once the ``with`` block exits (including on an exception).

    Yields
    ------
    None
    """
    start = time.perf_counter()
    logger.debug("stage=%s status=start", stage)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[stage] = elapsed
        logger.info("stage=%s status=done duration_s=%.3f", stage, elapsed)
