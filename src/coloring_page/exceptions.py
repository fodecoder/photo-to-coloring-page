"""Typed error hierarchy for every failure this package raises intentionally.

Before this module, the only custom exception in the codebase was
``pipeline.UnsupportedFormatError(ValueError)``, and everything else that
could go wrong (a missing checkpoint, an oversized input, an unknown
``--style``) surfaced as a bare builtin exception or an unhandled stack
trace. A caller embedding :func:`coloring_page.convert_image` in a service
needs to distinguish these failure modes programmatically (e.g. retry on
``WeightsMissingError`` after fetching weights, but not on
``UnsupportedImageError``) -- catching ``ColoringPageError`` is also the one
thing that reliably separates "this input/environment is the problem" from
an actual bug in this package.
"""

from __future__ import annotations


class ColoringPageError(Exception):
    """Base class for every error this package raises intentionally.

    Catching this (rather than ``Exception``) is how a caller -- the CLI,
    or a service embedding :func:`coloring_page.convert_image` -- tells
    "expected, typed failure" apart from an actual bug in this package.
    """


class UnsupportedImageError(ColoringPageError):
    """Raised for an input that isn't a usable image.

    Covers a bad file extension, a file OpenCV can't decode, and an
    in-memory ``np.ndarray`` that doesn't match the ``(H, W, 3)`` BGR
    ``uint8`` contract :func:`coloring_page.convert_image` requires.
    """


class ImageTooLargeError(ColoringPageError):
    """Raised when an input image exceeds ``Profile.max_pixels``.

    A resource limit, not a format problem: the image is perfectly valid,
    it's just larger than the caller configured this package to process,
    which matters when :func:`coloring_page.convert_image` is exposed to
    untrusted input in a service.
    """


class WeightsMissingError(ColoringPageError):
    """Raised when a required model checkpoint is not present on disk.

    See :mod:`coloring_page.weights` for where checkpoints are looked up.
    """


class WeightsChecksumError(ColoringPageError):
    """Raised when a checkpoint on disk does not match its pinned SHA256.

    Verified every time a checkpoint is loaded (not only when it is
    downloaded) -- a corrupted or tampered file must fail loudly, not
    silently produce degraded or unpredictable output.
    """


class EngineUnavailableError(ColoringPageError):
    """Raised for an unknown ``--style``/``Profile.style`` name.

    Also covers a style whose optional extra (``ml``, ``lineart``, ...)
    isn't installed, since that style is equally absent from
    ``engines.registry.ENGINES`` either way.
    """


class QualityGateError(ColoringPageError):
    """Raised by the CLI when ``--strict`` is set and a ``QualityReport`` fails.

    The library itself never raises this: :func:`coloring_page.convert_image`
    always returns a ``Result`` regardless of whether its ``report`` passed,
    so a caller can inspect quality and decide for itself. Only the CLI
    turns a failing report into a hard error.
    """
