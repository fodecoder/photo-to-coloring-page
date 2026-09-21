"""Tests for the typed error hierarchy."""

from __future__ import annotations

from coloring_page.exceptions import (
    ColoringPageError,
    EngineUnavailableError,
    ImageTooLargeError,
    QualityGateError,
    UnsupportedImageError,
    WeightsChecksumError,
    WeightsMissingError,
)


class TestExceptionHierarchy:
    def test_every_concrete_error_derives_from_base(self) -> None:
        concrete = [
            UnsupportedImageError,
            ImageTooLargeError,
            WeightsMissingError,
            WeightsChecksumError,
            EngineUnavailableError,
            QualityGateError,
        ]
        for exc_type in concrete:
            assert issubclass(exc_type, ColoringPageError)

    def test_base_derives_from_exception(self) -> None:
        assert issubclass(ColoringPageError, Exception)

    def test_instances_are_catchable_via_base(self) -> None:
        try:
            raise ImageTooLargeError("too big")
        except ColoringPageError as exc:
            assert str(exc) == "too big"
        else:
            raise AssertionError("expected ColoringPageError to catch ImageTooLargeError")
