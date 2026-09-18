"""Region-boundary conversion engine: segment into regions, draw where they meet.

Every other engine in this package answers "where is there a sharp
intensity gradient?" -- that is what an edge detector, by construction,
can answer. A coloring page needs a different question answered: "where
does one object end and another begin?" On an illustrated or painted
source image those two questions have different answers everywhere a
glow, a soft shadow, or a shaded gradient produces a real intensity
change that is not an object boundary. `docs/DIAGNOSIS.md` #5 measured
this directly: 36 parameter combinations of the classical gradient-based
engines topped out at F1 0.587, a structural ceiling, not a tuning
problem.

This engine instead segments the image into flat color regions first,
then draws only the boundaries *between regions* -- which are closed by
construction and cannot fire inside a smooth gradient with no region
boundary in it. Five explicit stages:

1. **flatten** -- ``cv2.ximgproc.l0Smooth`` removes small-scale texture
   and gradients while keeping sharp transitions sharp (the same
   motivation as ``chained.py``'s rolling guidance filter, at strength
   suited to color segmentation rather than edge chaining).
2. **segment** -- flat-color regions, as an integer label per pixel. Two
   measured methods (see :class:`RegionEngine`'s docstring for the
   comparison): mean-shift filtering's posterized output, or superpixels
   (SEEDS/SLIC).
3. **boundaries** -- pixels where two differently-labeled regions meet.
4. **filter** -- discards boundaries that are short (segmentation noise)
   or separate two regions of near-identical color (a segmentation
   artifact, not a real object edge) -- the second criterion is the one
   a gradient-based edge detector structurally cannot apply, since it
   has no notion of "region" to compare.
5. **redraw** -- reuses :func:`~coloring_page.postprocess.redraw_segments`.

``cartoon.py`` already does mean-shift region segmentation in this
project, but stops at thresholding the segmented image's morphological
gradient into a raw mask -- no texture-flattening pre-step, no
inter-region contrast filter, no uniform-width redraw. This engine is
what that approach looks like carried through properly; see
:class:`RegionEngine`'s docstring for how it actually measures against
the gradient-based engines despite that.
"""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np

from coloring_page.engines.base import ConversionEngine, DebugSink
from coloring_page.pipeline import derive_kernel_size
from coloring_page.postprocess import redraw_segments

#: 4-connected neighbor offsets used to find where region labels change.
_NEIGHBOR_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _flatten(image: np.ndarray, *, lambda_: float, kappa: float) -> np.ndarray:
    """L0-smooth ``image``: flatten texture and gradients, keep sharp transitions.

    Parameters
    ----------
    image : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
    lambda_ : float
        Smooth-term weight; larger values flatten more aggressively.
    kappa : float
        Growth factor for the gradient data term across iterations.

    Returns
    -------
    np.ndarray
        Flattened BGR image, same shape and dtype as ``image``.
    """
    smoothed: np.ndarray = cv2.ximgproc.l0Smooth(image, None, lambda_, kappa)
    return smoothed


def _labels_from_colors(posterized: np.ndarray, *, quantize_step: int) -> np.ndarray:
    """Group pixels sharing a (quantized) color into integer labels.

    Grouping is by color identity only, not spatial connectivity: two
    disjoint patches of the same posterized color (e.g. sky visible on
    both sides of a rooftop) reasonably share a region for this engine's
    purposes -- the boundary/contrast stages downstream only need each
    pixel's region's mean color, which is identical either way, and
    skipping connected-components labeling keeps this a single vectorized
    pass instead of one ``cv2.connectedComponents`` call per unique color.

    ``pyrMeanShiftFiltering`` drives most same-region pixels to the exact
    same color, but not perfectly -- a residual gradient of a few
    intensity levels can survive inside a large flat region. Quantizing
    first merges that residual noise into one label instead of
    fragmenting the region into many near-duplicate ones.

    Parameters
    ----------
    posterized : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``, already
        posterized (e.g. by mean-shift filtering).
    quantize_step : int
        Round each channel down to a multiple of this before grouping.

    Returns
    -------
    np.ndarray
        Integer label image, shape ``(H, W)``, dtype ``int32``.
    """
    quantized = (posterized // quantize_step).astype(np.int32)
    flat = quantized.reshape(-1, quantized.shape[2])
    _, inverse = np.unique(flat, axis=0, return_inverse=True)
    return inverse.reshape(posterized.shape[:2]).astype(np.int32)


def segment_mean_shift(
    image: np.ndarray, *, sp: int, sr: int, max_level: int = 2, quantize_step: int = 4
) -> np.ndarray:
    """Segment ``image`` into regions via mean-shift filtering.

    Parameters
    ----------
    image : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8`` (should already
        be flattened -- see :func:`_flatten`).
    sp : int
        Spatial window radius, in pixels. Scale with resolution via
        :func:`~coloring_page.pipeline.derive_kernel_size`.
    sr : int
        Color window radius. Not a pixel measurement, so not
        resolution-dependent -- this is a distance in BGR color space.
    max_level : int, optional
        Pyramid levels for the mean-shift filter, by default 2.
    quantize_step : int, optional
        Passed to :func:`_labels_from_colors`, by default 4.

    Returns
    -------
    np.ndarray
        Integer label image, shape ``(H, W)``, dtype ``int32``.
    """
    posterized = cv2.pyrMeanShiftFiltering(image, sp, sr, maxLevel=max_level)
    return _labels_from_colors(posterized, quantize_step=quantize_step)


def segment_superpixels(
    image: np.ndarray,
    *,
    algorithm: Literal["seeds", "slic"] = "seeds",
    num_superpixels: int = 500,
    num_iterations: int = 10,
) -> np.ndarray:
    """Segment ``image`` into regions via superpixel over-segmentation.

    Unlike :func:`segment_mean_shift`, this gives direct control over how
    many regions result, independent of any color-distance threshold --
    useful when mean-shift's ``sr`` either over- or under-segments a
    given image. Superpixels alone *over*-segment on purpose (that's the
    point of the technique); this engine relies on the ``filter`` stage's
    contrast check to merge same-object superpixels back together at the
    boundary-drawing step, rather than merging labels explicitly.

    Parameters
    ----------
    image : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
    algorithm : {"seeds", "slic"}, optional
        Which superpixel algorithm to use, by default ``"seeds"``.
    num_superpixels : int, optional
        Target superpixel count, by default 500. A *count*, not a pixel
        measurement, so not derived from the working resolution the way
        a kernel size would be -- proportionally more superpixels are
        wanted on a larger image to keep the same average superpixel
        *area*, but this engine does not currently auto-scale it; users
        converting much larger or smaller images than
        :data:`~coloring_page.pipeline.DEFAULT_WORKING_DIMENSION` may
        want to adjust it.
    num_iterations : int, optional
        Refinement iterations, by default 10.

    Returns
    -------
    np.ndarray
        Integer label image, shape ``(H, W)``, dtype ``int32``.
    """
    height, width, channels = image.shape
    if algorithm == "seeds":
        seeds = cv2.ximgproc.createSuperpixelSEEDS(
            width, height, channels, num_superpixels, num_levels=4
        )
        seeds.iterate(image, num_iterations)
        labels = seeds.getLabels()
    elif algorithm == "slic":
        region_size = max(4, round((height * width / num_superpixels) ** 0.5))
        slic = cv2.ximgproc.createSuperpixelSLIC(
            image, algorithm=cv2.ximgproc.SLICO, region_size=region_size
        )
        slic.iterate(num_iterations)
        labels = slic.getLabels()
    else:
        raise ValueError(f"Unknown superpixel algorithm {algorithm!r}; expected 'seeds' or 'slic'.")
    return labels.astype(np.int32)


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels 4-adjacent to a differently-labeled pixel.

    Tracing where the label map itself changes gives closed-by-construction
    region boundaries directly, without picking an arbitrary gradient
    threshold -- see :func:`gradient_boundaries` for the alternative this
    was measured against.

    Parameters
    ----------
    labels : np.ndarray
        Integer label image, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Boolean mask, same shape as ``labels``.
    """
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    return boundary


def gradient_boundaries(lab_image: np.ndarray, *, percentile: float) -> np.ndarray:
    """Boolean mask of high-gradient pixels in Lab space, as a fraction of gradient magnitude.

    Alternative to :func:`_label_boundaries` that skips segmentation
    labels entirely and thresholds the Lab-space gradient directly (Lab,
    not grayscale, because two colors of equal luminance but different
    hue are a real object boundary that a grayscale gradient loses). Note
    this operates on the *flattened* image, before segmentation, so its
    raw output is identical regardless of which ``segmentation`` method
    is chosen -- only the downstream contrast-filtering step (which does
    use the segmentation labels) makes the two segmentation methods'
    final results differ when paired with this boundary detector.

    Measured with ``scripts/compare.py`` against the label-transition
    approach (:func:`_label_boundaries`), paired with mean-shift
    segmentation, on this project's 3 reference images: this won on 2 of
    3 (``f1_normalized`` 0.485 vs 0.439, and 0.298 vs 0.261), and lost on
    the third (0.530 vs 0.592) -- close enough, and inconsistent enough
    across images, that neither is a clear default on this evidence
    alone; :class:`RegionEngine` defaults to this one since it wins the
    majority, but the label-transition alternative measurably suits some
    images better and is a reasonable first thing to try if this one
    underperforms on a given photo.

    Parameters
    ----------
    lab_image : np.ndarray
        Image in Lab color space, shape ``(H, W, 3)``.
    percentile : float
        Percentile of gradient magnitude used as the keep threshold.

    Returns
    -------
    np.ndarray
        Boolean mask, shape ``(H, W)``.
    """
    lab_float = lab_image.astype(np.float32)
    gx = cv2.Sobel(lab_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab_float, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.sum(gx**2, axis=2) + np.sum(gy**2, axis=2))
    threshold = np.percentile(magnitude, percentile)
    return cast(np.ndarray, magnitude >= threshold)


def _mean_lab_per_label(lab_image: np.ndarray, labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Mean Lab color of each label.

    Parameters
    ----------
    lab_image : np.ndarray
        Image in Lab color space, shape ``(H, W, 3)``.
    labels : np.ndarray
        Integer label image, shape ``(H, W)``, values in ``[0, num_labels)``.
    num_labels : int
        Number of distinct labels.

    Returns
    -------
    np.ndarray
        Shape ``(num_labels, 3)``, mean Lab color per label.
    """
    flat_lab = lab_image.reshape(-1, 3).astype(np.float64)
    flat_labels = labels.reshape(-1)
    sums = np.zeros((num_labels, 3), dtype=np.float64)
    counts = np.zeros(num_labels, dtype=np.float64)
    np.add.at(sums, flat_labels, flat_lab)
    np.add.at(counts, flat_labels, 1)
    return sums / np.maximum(counts, 1)[:, None]


def filter_by_contrast(
    boundary: np.ndarray, labels: np.ndarray, lab_image: np.ndarray, *, min_contrast: float
) -> np.ndarray:
    """Drop boundary pixels whose two adjacent regions are near-identical in color.

    This is the criterion a gradient-based edge detector cannot apply --
    it has no notion of "region" to compare, only a local pixel
    neighborhood. A boundary between two regions of nearly the same mean
    Lab color is a segmentation artifact (the flatten/segment stages
    split one real object into two labels), not an object edge, and
    should be dropped rather than redrawn.

    Parameters
    ----------
    boundary : np.ndarray
        Boolean mask from :func:`_label_boundaries` or
        :func:`gradient_boundaries`.
    labels : np.ndarray
        Integer label image, shape ``(H, W)``, same shape as ``boundary``.
    lab_image : np.ndarray
        Image in Lab color space, shape ``(H, W, 3)``.
    min_contrast : float
        Minimum Euclidean Lab distance between two adjacent regions'
        mean colors for a boundary pixel between them to survive.

    Returns
    -------
    np.ndarray
        Boolean mask, same shape as ``boundary``, a subset of it.
    """
    num_labels = int(labels.max()) + 1
    mean_lab = _mean_lab_per_label(lab_image, labels, num_labels)

    height, width = labels.shape
    ys, xs = np.nonzero(boundary)
    keep = np.zeros(len(ys), dtype=bool)
    own_lab = mean_lab[labels[ys, xs]]

    for dy, dx in _NEIGHBOR_OFFSETS:
        ny, nx = ys + dy, xs + dx
        valid = (0 <= ny) & (ny < height) & (0 <= nx) & (nx < width)
        neighbor_lab = np.zeros_like(own_lab)
        neighbor_lab[valid] = mean_lab[labels[ny[valid], nx[valid]]]
        contrast = np.linalg.norm(own_lab - neighbor_lab, axis=1)
        keep |= valid & (contrast >= min_contrast)

    result = np.zeros(boundary.shape, dtype=bool)
    result[ys[keep], xs[keep]] = True
    return result


def _mask_to_line_art(mask: np.ndarray) -> np.ndarray:
    """Render a boolean mask in this project's convention (0=ink, 255=background)."""
    return np.where(mask, 0, 255).astype(np.uint8)


class RegionEngine(ConversionEngine):
    """Draw region boundaries instead of intensity-gradient edges.

    Segmentation method comparison (measured with ``scripts/compare.py``
    on this project's 3 reference pairs, boundary detection fixed to
    ``"gradient"``): mean-shift segmentation decisively beat SEEDS/SLIC
    superpixels, not just narrowly -- ``f1_normalized`` of 0.44-0.59 vs
    0.00-0.14 across the 3 images. Superpixels are available
    (``segmentation="seeds"``/``"slic"``) but were not competitive here;
    a fixed target region count, independent of any color-distance
    threshold, could still suit an image where mean-shift's ``sr``
    badly over- or under-merges, but that has not been observed on this
    project's reference set.

    Measured against ``canny``/``chained`` on the same 3 pairs: this
    engine does **not** clear this project's original bar of beating
    both on at least 2 of 3 -- it wins on 1 of 3 (the pair without a
    known aspect-ratio mismatch is the one it loses, narrowly, to both).
    It does have a real, measured advantage in ink-admissibility (in the
    3-7% band on all 3 images here, vs 0 of 3 for ``canny`` -- corrected
    from an earlier miscount of this same data before this docstring
    settled on ``"gradient"`` as the default), which is the property the
    whole region-boundary approach was meant to
    improve on -- see ``docs/DIAGNOSIS.md`` #5. Shipped as a selectable
    ``--style`` for that reason, not as a proven replacement for the
    gradient-based engines; Phase 5's default-style decision should
    weigh this shortfall, not assume this phase's original hypothesis
    was confirmed.

    See :func:`gradient_boundaries` for the boundary-detection comparison
    (``"gradient"`` vs ``"labels"``).
    """

    name = "region"

    def __init__(
        self,
        l0_lambda: float = 0.02,
        l0_kappa: float = 2.0,
        segmentation: Literal["mean_shift", "seeds", "slic"] = "mean_shift",
        color_radius: int = 20,
        num_superpixels: int = 500,
        boundary_detection: Literal["labels", "gradient"] = "gradient",
        gradient_percentile: float = 94.0,
        min_contrast: float = 8.0,
        polyline_epsilon: float = 1.2,
    ) -> None:
        """Store the flatten/segment/boundary/filter stage parameters.

        Parameters
        ----------
        l0_lambda : float, optional
            :func:`_flatten`'s smooth-term weight, by default 0.02.
        l0_kappa : float, optional
            :func:`_flatten`'s gradient-term growth factor, by default 2.0.
        segmentation : {"mean_shift", "seeds", "slic"}, optional
            Which :func:`segment_mean_shift`/:func:`segment_superpixels`
            variant to use, by default ``"mean_shift"`` -- see the class
            docstring for the measured comparison.
        color_radius : int, optional
            ``sr`` passed to :func:`segment_mean_shift` when
            ``segmentation="mean_shift"``, by default 20. A color-space
            distance, not a pixel measurement, so not resolution-scaled.
        num_superpixels : int, optional
            Passed to :func:`segment_superpixels` when ``segmentation``
            is ``"seeds"`` or ``"slic"``, by default 500.
        boundary_detection : {"labels", "gradient"}, optional
            Which of :func:`_label_boundaries`/:func:`gradient_boundaries`
            to use, by default ``"gradient"`` -- see
            :func:`gradient_boundaries` for the measured comparison.
        gradient_percentile : float, optional
            Passed to :func:`gradient_boundaries` when
            ``boundary_detection="gradient"``, by default 94.0.
        min_contrast : float, optional
            Passed to :func:`filter_by_contrast`, by default 8.0 (a Lab
            Euclidean distance).
        polyline_epsilon : float, optional
            ``cv2.approxPolyDP`` tolerance used to smooth each boundary
            chain before redrawing, in pixels, by default 1.2.
        """
        self.l0_lambda = l0_lambda
        self.l0_kappa = l0_kappa
        self.segmentation = segmentation
        self.color_radius = color_radius
        self.num_superpixels = num_superpixels
        self.boundary_detection = boundary_detection
        self.gradient_percentile = gradient_percentile
        self.min_contrast = min_contrast
        self.polyline_epsilon = polyline_epsilon

    def convert(
        self, image: np.ndarray, *, line_thickness: int = 1, debug: DebugSink | None = None
    ) -> np.ndarray:
        """Segment into regions, then draw only the boundaries between them.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        working_dimension = max(image.shape[:2])

        flattened = _flatten(image, lambda_=self.l0_lambda, kappa=self.l0_kappa)
        if debug is not None:
            debug.save("flattened", flattened)

        if self.segmentation == "mean_shift":
            spatial_radius = derive_kernel_size(
                working_dimension, fraction=0.01, min_value=5, odd=False
            )
            labels = segment_mean_shift(flattened, sp=spatial_radius, sr=self.color_radius)
        else:
            labels = segment_superpixels(
                flattened, algorithm=self.segmentation, num_superpixels=self.num_superpixels
            )
        if debug is not None:
            debug.save("segments", _mask_to_line_art(_label_boundaries(labels)))

        lab = cv2.cvtColor(flattened, cv2.COLOR_BGR2LAB)
        if self.boundary_detection == "labels":
            boundary = _label_boundaries(labels)
        elif self.boundary_detection == "gradient":
            boundary = gradient_boundaries(lab, percentile=self.gradient_percentile)
        else:
            raise ValueError(
                f"Unknown boundary_detection {self.boundary_detection!r}; "
                "expected 'labels' or 'gradient'."
            )
        if debug is not None:
            debug.save("boundaries_raw", _mask_to_line_art(boundary))

        filtered = filter_by_contrast(boundary, labels, lab, min_contrast=self.min_contrast)
        if debug is not None:
            debug.save("boundaries_filtered", _mask_to_line_art(filtered))

        thinned: np.ndarray = np.zeros(filtered.shape, dtype=np.uint8)
        if filtered.any():
            thinned = cv2.ximgproc.thinning((filtered * 255).astype(np.uint8))

        min_length = derive_kernel_size(working_dimension, fraction=0.014, min_value=10, odd=False)
        contours, _ = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        segments = [c for c in contours if cv2.arcLength(c, closed=False) >= min_length]

        canvas = redraw_segments(
            segments,
            image.shape[:2],
            polyline_epsilon=self.polyline_epsilon,
            line_thickness=line_thickness,
        )
        if debug is not None:
            debug.save("redraw", canvas)
        return canvas
