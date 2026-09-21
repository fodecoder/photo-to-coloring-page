"""Convert photos into printable black-and-white coloring pages.

Public API::

    from coloring_page import convert_image, Profile

    result = convert_image(Path("photo.jpg"), profile=Profile(detail="child"))
    result.save_svg(Path("out.svg"))

See :func:`convert_image`, :class:`Profile`, and :class:`Result` for the
full contract. The CLI (``coloring_page.cli``) is a thin wrapper over this
same API.
"""

from __future__ import annotations

from coloring_page.api import convert_image
from coloring_page.exceptions import (
    ColoringPageError,
    EngineUnavailableError,
    ImageTooLargeError,
    QualityGateError,
    UnsupportedImageError,
    WeightsChecksumError,
    WeightsMissingError,
)
from coloring_page.page import PageSpec
from coloring_page.profile import Profile
from coloring_page.result import Result
from coloring_page.validate import QualityReport

__version__ = "0.1.0"

__all__ = [
    "convert_image",
    "Profile",
    "PageSpec",
    "Result",
    "QualityReport",
    "ColoringPageError",
    "UnsupportedImageError",
    "ImageTooLargeError",
    "WeightsMissingError",
    "WeightsChecksumError",
    "EngineUnavailableError",
    "QualityGateError",
]
