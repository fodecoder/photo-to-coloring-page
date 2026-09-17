"""XDoG (eXtended Difference-of-Gaussians) conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine
from coloring_page.pipeline import (
    DEFAULT_WORKING_DIMENSION,
    derive_kernel_size,
    remove_short_strokes,
    smooth_preserving_edges,
)


class XDoGEngine(ConversionEngine):
    """Produce line art with an eXtended Difference-of-Gaussians filter.

    XDoG subtracts two differently-blurred copies of the image to find
    edges (like a standard DoG filter), then applies a sharpened
    tanh-based threshold to that difference. Compared to ``canny``, it
    tends to produce more consistent, artistic-looking outlines and
    handles gradual shading transitions more gracefully.
    """

    name = "xdog"

    def __init__(
        self,
        sigma: float = 1.0,
        k: float = 1.6,
        p: float = 25.0,
        epsilon: float = 0.1,
        phi: float = 40.0,
    ) -> None:
        """Store the XDoG filter parameters used on every conversion.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of the smaller Gaussian blur, by default
            1.0. Tuned for :data:`coloring_page.pipeline.DEFAULT_WORKING_DIMENSION`;
            scaled proportionally to the actual image size in
            :meth:`convert` so it stays meaningful at other working
            resolutions.
        k : float, optional
            Multiplier applied to ``sigma`` for the larger Gaussian blur, by
            default 1.6.
        p : float, optional
            Sharpening factor applied to the difference-of-Gaussians before
            thresholding, by default 25.0. Higher values give sharper,
            higher-contrast lines.
        epsilon : float, optional
            Threshold on the sharpened difference-of-Gaussians value at
            or above which a pixel is treated as background (paper), by
            default 0.1. In flat regions this value tracks the raw pixel
            intensity (0-1), but real photographed pages rarely reach
            very high normalized brightness even on their paper/light
            background (lighting falloff, JPEG compression, illustration
            covering most of the page) -- an epsilon near the "should be
            pure white" end of that range leaves nearly everything below
            threshold and turns almost the whole page to ink. This
            default was set empirically against
            ``docs/starting-image.jpeg`` rather than a fixed high
            constant, to keep ink coverage in the project's target range
            despite variation in real-world photo brightness.
        phi : float, optional
            Sharpness of the soft threshold's tanh transition below
            ``epsilon``, by default 40.0. Higher values push the
            transition closer to a hard step, producing solid black
            rather than a gray gradient near the threshold.
        """
        self.sigma = sigma
        self.k = k
        self.p = p
        self.epsilon = epsilon
        self.phi = phi

    def convert(self, image: np.ndarray, *, line_thickness: int = 1) -> np.ndarray:
        """Compute an extended difference-of-Gaussians edge map.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = smooth_preserving_edges(gray_u8)
        gray = denoised.astype(np.float64) / 255.0

        working_dimension = max(gray.shape[:2])
        effective_sigma = self.sigma * (working_dimension / DEFAULT_WORKING_DIMENSION)

        g1 = cv2.GaussianBlur(gray, (0, 0), effective_sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), effective_sigma * self.k)
        # Sharp DoG (Winnemoeller et al. 2012): (1+p)*g1 - p*g2, not
        # g1 - p*(g1-g2) -- that earlier form expands to (1-p)*g1 + p*g2,
        # an inverted sign that, combined with a negative-domain epsilon,
        # left nearly every pixel classified as background.
        diff = (1.0 + self.p) * g1 - self.p * g2

        # Extended thresholding: pixels at/above epsilon become white
        # (paper); pixels below it are sharpened toward black with a
        # tanh whose steepness is controlled by phi, rather than a bare
        # tanh(diff - epsilon) which is nearly linear over this range
        # and rarely reaches true black.
        xdog = np.where(
            diff >= self.epsilon,
            1.0,
            1.0 + np.tanh(self.phi * (diff - self.epsilon)),
        )
        xdog = np.clip(xdog, 0.0, 1.0)
        lines = (xdog * 255).astype(np.uint8)
        min_extent = derive_kernel_size(working_dimension, fraction=0.012, min_value=3, odd=False)
        lines = remove_short_strokes(lines, min_extent=min_extent)

        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            lines = cv2.erode(lines, kernel, iterations=1).astype(np.uint8)

        return lines
