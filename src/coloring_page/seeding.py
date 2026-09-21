"""Deterministic seeding for reproducible conversions.

A given ``(image, Profile)`` pair should produce the same ``Drawing``
every time, including through the ML engines' stochastic operations (e.g.
SAM 2's mask sampling). Nothing seeded ``numpy``/``torch`` before this
module existed.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int | None) -> None:
    """Seed every source of randomness this package's engines can use.

    Parameters
    ----------
    seed : int | None
        The seed to apply. If None, this function does nothing -- a
        caller who explicitly wants nondeterministic behavior (e.g. to
        sample several different outputs for the same input) passes
        ``Profile(seed=None)``.

    Notes
    -----
    Seeding does not cross process boundaries: a caller parallelizing
    conversions across worker processes (see :mod:`coloring_page.batch`)
    must call this again at the start of each worker task.

    ``torch.use_deterministic_algorithms(True)`` is attempted when torch
    is importable; some operations have no deterministic CUDA kernel and
    raise ``RuntimeError`` when actually invoked under that setting, which
    is logged as a warning rather than propagated, since a caller running
    on CPU or with an engine that avoids that operation is unaffected.
    Full CUDA determinism additionally requires the ``CUBLAS_WORKSPACE_CONFIG``
    environment variable to be set by the caller before the process starts
    (torch's own requirement -- see the README).
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        logger.warning("Could not enable fully deterministic algorithms: %s", exc)
