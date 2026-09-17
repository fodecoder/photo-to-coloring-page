"""XDoG (eXtended Difference-of-Gaussians) conversion engine."""

from __future__ import annotations

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine


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
        sigma: float = 0.5,
        k: float = 1.6,
        p: float = 20.0,
        epsilon: float = -0.1,
    ) -> None:
        """Store the XDoG filter parameters used on every conversion.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of the smaller Gaussian blur, by default 0.5.
        k : float, optional
            Multiplier applied to ``sigma`` for the larger Gaussian blur, by
            default 1.6.
        p : float, optional
            Sharpening factor applied to the difference-of-Gaussians before
            thresholding, by default 20.0. Higher values give sharper,
            higher-contrast lines.
        epsilon : float, optional
            Threshold (in normalized intensity units) below which a pixel is
            treated as background, by default -0.1.
        """
        self.sigma = sigma
        self.k = k
        self.p = p
        self.epsilon = epsilon

    def convert(self, image: np.ndarray, *, line_thickness: int = 1) -> np.ndarray:
        """Compute an extended difference-of-Gaussians edge map.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

        g1 = cv2.GaussianBlur(gray, (0, 0), self.sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), self.sigma * self.k)
        diff = g1 - self.p * (g1 - g2)

        # Extended thresholding: pixels above epsilon become white (paper),
        # pixels at or below it are sharpened toward black (line) with tanh.
        xdog = np.where(
            diff >= self.epsilon,
            1.0,
            1.0 + np.tanh((diff - self.epsilon)),
        )
        xdog = np.clip(xdog, 0.0, 1.0)
        lines = (xdog * 255).astype(np.uint8)

        if line_thickness > 1:
            kernel = np.ones((line_thickness, line_thickness), np.uint8)
            lines = cv2.erode(lines, kernel, iterations=1)

        return lines
