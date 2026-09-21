"""Production SAM-segmentation + detail-line conversion engine (``--style lineart``).

Every other engine in this package (including ``region``, its closest
relative) still answers a *local* question -- "is there a sharp gradient or
color change here?" -- one pixel neighborhood at a time. That structurally
caps precision: texture (foliage, paper grain, a patterned shirt) produces
exactly the same local signal as a real object edge. This engine instead
asks a *global* question first: SAM 2's automatic mask generator partitions
the whole photo into semantic regions in one pass (Stage A), so what
survives as "a boundary" is decided by object identity, not local contrast.
Region boundaries alone lose genuinely fine internal detail no
segmentation model resolves (an eye, a mouth, a fold, a deliberate
pattern) -- a purpose-built line-extraction network recovers that
separately (Stage B). The two candidate sources are then pruned against a
physical, print-size-relative notion of "small enough to matter, big
enough to color" (Stage C) and turned into smooth, deduplicated vector
paths (Stage D). See ``docs/DIAGNOSIS.md`` #5-#6 and
``docs/PRODUCTION-PROMPTS.md``'s Prompt 3 for the full design rationale
this engine implements.

Requires the ``lineart`` extra (``pip install -e ".[lineart]"``) for
``torch``, ``sam2``, and ``controlnet_aux``, plus two manually downloaded
checkpoints -- neither bundled with this project. See
``scripts/fetch_weights.py`` and the README's "Extending with an ML
engine" section.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

# Unused directly at module level -- imported here (rather than only
# inside the functions that use them) purely so importing this module
# raises ImportError whenever any of the `lineart` extra's dependencies
# is missing, matching informative_drawings.py's unconditional `import
# torch`. registry.py's try/except ImportError around this module's
# import is what keeps "lineart" out of `ENGINES` (and so out of
# `--style`'s choices) until `pip install -e ".[lineart]"` has actually
# been run -- without this, a missing sam2/controlnet_aux would only
# surface as an ImportError deep inside convert(), not as the style
# simply not being offered.
import controlnet_aux  # noqa: F401
import cv2
import numpy as np
import sam2  # noqa: F401
from scipy.spatial.distance import directed_hausdorff

from coloring_page.drawing import Drawing, paths_from_mask, paths_from_point_chains, rasterize
from coloring_page.drawing import Path as DrawingPath
from coloring_page.engines._lineart_seg import build_label_map, generate_masks, load_mask_generator
from coloring_page.engines.base import ConversionEngine, DebugSink

# Reused rather than reimplemented: both this engine's segmentation and
# detail-line stages need the exact same "short side, multiple of 64,
# aspect-ratio preserved" working-resolution convention this function
# already implements for informative_drawings -- see its own docstring for
# why (it matches controlnet_aux's own resize convention, which the
# lllyasviel/Annotators-family weights this engine's Stage B also draws
# from were trained/exercised against).
from coloring_page.engines.informative_drawings import _resize_short_side
from coloring_page.exceptions import WeightsMissingError
from coloring_page.postprocess import hysteresis_centerline, normalize_percentile
from coloring_page.weights import resolve_weights_path, verify_checksum

#: 4-connected neighbor offsets used to find where region labels change.
_NEIGHBOR_OFFSETS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

#: Env vars pointing at a local weights file, in place of the default
#: lookup locations below.
WEIGHTS_ENV_VAR = "COLORING_PAGE_LINEART_SEGMENTATION_WEIGHTS"
DETAIL_WEIGHTS_ENV_VAR = "COLORING_PAGE_LINEART_DETAIL_WEIGHTS"

DEFAULT_SEGMENTATION_WEIGHTS_PATH = (
    Path.home() / ".cache" / "coloring_page" / "sam2.1_hiera_small.pt"
)
_LOCAL_SEGMENTATION_WEIGHTS_PATH = Path("weights") / "sam2.1_hiera_small.pt"
#: Kept as "netG.pth" (not renamed, unlike the other checkpoints this
#: project fetches) since controlnet_aux's own LineartAnimeDetector.from_
#: pretrained() expects to find the checkpoint under exactly this name in
#: whatever directory it's pointed at -- see scripts/fetch_weights.py.
DEFAULT_DETAIL_WEIGHTS_PATH = Path.home() / ".cache" / "coloring_page" / "netG.pth"
_LOCAL_DETAIL_WEIGHTS_PATH = Path("weights") / "netG.pth"

_SEG_WEIGHTS_HELP = (
    "SAM 2.1 Hiera-small checkpoint not found at {path}.\n"
    "Run 'python scripts/fetch_weights.py --filename sam2.1_hiera_small.pt' "
    "(requires the 'lineart' extra's huggingface_hub dependency), or "
    "download it manually from "
    "https://huggingface.co/facebook/sam2.1-hiera-small.\n"
    f"Save it to {_LOCAL_SEGMENTATION_WEIGHTS_PATH} (relative to the "
    f"current directory), {DEFAULT_SEGMENTATION_WEIGHTS_PATH}, or set the "
    f"{WEIGHTS_ENV_VAR} environment variable to wherever you saved it.\n"
    "SAM 2 is Apache-2.0 licensed (Copyright Meta Platforms, Inc.); see "
    "THIRD_PARTY_LICENSES.md."
)

_DETAIL_WEIGHTS_HELP = (
    "lineart_anime detail-line checkpoint (netG.pth) not found at {path}.\n"
    "Run 'python scripts/fetch_weights.py --filename netG.pth --dest "
    f"{_LOCAL_DETAIL_WEIGHTS_PATH}' (requires the 'lineart' extra's "
    "huggingface_hub dependency).\n"
    f"Save it to {_LOCAL_DETAIL_WEIGHTS_PATH} (relative to the current "
    f"directory), {DEFAULT_DETAIL_WEIGHTS_PATH}, or set the "
    f"{DETAIL_WEIGHTS_ENV_VAR} environment variable to wherever you saved it.\n"
    "Do not substitute weights/informative_drawings.pth here: that "
    "checkpoint produces hatching/shading, not outline linework -- see "
    "docs/DIAGNOSIS.md #3. See THIRD_PARTY_LICENSES.md for the "
    "controlnet_aux lineart_anime checkpoint's license."
)


def _resolve_lineart_weights_path(
    explicit_path: str | Path | None, filename: str, env_var: str, local_path: Path
) -> Path:
    """Pick a weights file to use -- see ``coloring_page.weights.resolve_weights_path``.

    Shared by this engine's two independent checkpoints (segmentation,
    detail-line) rather than duplicated per-checkpoint.
    """
    return resolve_weights_path(
        filename, explicit_path=explicit_path, legacy_env_var=env_var, local_path=local_path
    )


def _resolve_device(device: Literal["auto", "cpu", "cuda", "mps"]) -> str:
    """Resolve ``"auto"`` to a concrete torch device string; pass the rest through.

    No device-resolution helper exists elsewhere in this project yet --
    every current ML engine (``anime2sketch``, ``informative_drawings``)
    hardcodes CPU inference. This engine is the first that needs GPU
    support (SAM 2's automatic mask generator is not practically fast
    enough on CPU for interactive use), so the helper lives here rather
    than being retrofitted onto engines that don't use it.
    """
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mm_to_px(value_mm: float, *, dpi: float) -> float:
    """Convert a physical length in millimeters, at ``dpi``, to pixels.

    Standalone so it's unit-testable without constructing a full engine --
    see the module docstring for why this engine's public parameters are
    expressed in millimeters at print size rather than working-resolution
    pixels: "how small a region can be for a child to still color it" is a
    physical quantity, independent of whatever resolution a given photo
    happens to be detected at.
    """
    return value_mm / 25.4 * dpi


def _mm2_to_px2(value_mm2: float, *, dpi: float) -> float:
    """Convert a physical area in square millimeters, at ``dpi``, to square pixels."""
    px_per_mm = dpi / 25.4
    return value_mm2 * px_per_mm**2


def _build_mask_generator(weights_path: Path, *, device: str) -> Any:
    """Lazily construct a SAM 2 automatic mask generator from a checkpoint on disk."""
    if not weights_path.exists():
        raise WeightsMissingError(_SEG_WEIGHTS_HELP.format(path=weights_path))
    verify_checksum(weights_path, "sam2.1_hiera_small.pt")
    return load_mask_generator(weights_path, device=device)


def _build_detail_model(weights_path: Path, *, device: str) -> Any:
    """Lazily construct the detail-line model (controlnet_aux's ``lineart_anime``).

    Not exercised by this project's test suite (see
    ``tests/engines/test_lineart.py``'s stub-injection pattern) or by any
    committed run against real weights yet -- this integration should be
    validated end-to-end against the actually-installed ``controlnet_aux``
    version before this engine's pre-merge gates are considered satisfied
    (see the "Extending with an ML engine" section of the README).
    """
    if not weights_path.exists():
        raise WeightsMissingError(_DETAIL_WEIGHTS_HELP.format(path=weights_path))
    verify_checksum(weights_path, "netG.pth")
    from controlnet_aux import LineartAnimeDetector
    from PIL import Image

    detector = LineartAnimeDetector.from_pretrained(str(weights_path.parent)).to(device)

    def _run(image_rgb: np.ndarray) -> np.ndarray:
        result = detector(Image.fromarray(image_rgb), output_type="np")
        return np.asarray(result)

    return _run


def _segment(image: np.ndarray, mask_generator: Any, *, resolution: int) -> np.ndarray:
    """Stage A: partition ``image`` into a semantic, no-overlap/no-gap label map.

    Runs SAM 2's automatic mask generator at its own working resolution
    (short side rounded to a multiple of 64, aspect ratio preserved), not
    at whatever resolution ``image`` already is -- the working resolution
    an engine happens to receive an image at (see
    :data:`~coloring_page.pipeline.DEFAULT_WORKING_DIMENSION`) is
    unrelated to the resolution a segmentation model was trained/tuned to
    run well at. The label map is resized back (nearest-neighbor, to
    preserve hard label boundaries rather than blending them) to
    ``image``'s own resolution so downstream stages don't need to track
    two different working sizes.

    Parameters
    ----------
    image : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
    mask_generator : Any
        As returned by :func:`_build_mask_generator`
        (``sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator``), or
        any object exposing a compatible ``.generate`` method -- tests
        inject a stub here.
    resolution : int
        Target short-side resolution SAM runs at.

    Returns
    -------
    np.ndarray
        Integer label image, same ``(H, W)`` as ``image``, dtype ``int32``.
    """
    height, width = image.shape[:2]
    resized = _resize_short_side(image, resolution)
    masks = generate_masks(mask_generator, resized)
    labels = build_label_map(masks, resized.shape[:2])
    return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)


def _detail_lines(image: np.ndarray, detail_model: Any, *, resolution: int) -> np.ndarray:
    """Stage B: extract fine internal linework a region partition can't see.

    Runs a purpose-built line-extraction network (controlnet_aux's
    ``lineart_anime``) over ``image`` at its own working resolution,
    returning its raw soft response, contrast-stretched but *not yet
    binarized* -- binarization happens in :func:`_merge_and_prune`, using
    ``postprocess.hysteresis_centerline`` only, never
    ``postprocess.gradient_edges`` ("nms"): the latter traces a stroke's
    two edges instead of its centerline, doubling every real line
    (``docs/DIAGNOSIS.md`` #2) -- exactly the bug this project already
    fixed once for the ``informative_drawings``/``gated`` engines'
    default. Deliberately does *not* reuse
    ``weights/informative_drawings.pth``'s "fine" checkpoint here: that
    checkpoint produces hatching/shading, the wrong style for this stage
    (``docs/DIAGNOSIS.md`` #3).

    Parameters
    ----------
    image : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
    detail_model : Any
        A callable ``detail_model(image_rgb) -> np.ndarray`` returning a
        single-channel (or 3-channel, converted to grayscale here) soft
        response where lower values are more ink-like, or a stub with the
        same interface -- tests inject a stub here.
    resolution : int
        Target short-side resolution the model runs at.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` image, same ``(H, W)`` as ``image``,
        contrast-stretched via ``postprocess.normalize_percentile``.
    """
    height, width = image.shape[:2]
    resized = _resize_short_side(image, resolution)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    response = np.asarray(detail_model(image_rgb))
    if response.ndim == 3:
        response = cv2.cvtColor(response, cv2.COLOR_RGB2GRAY)
    normalized = normalize_percentile(response.astype(np.uint8))
    return cv2.resize(normalized, (width, height), interpolation=cv2.INTER_LINEAR)


def _mean_lab_per_label(lab_image: np.ndarray, labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Mean Lab color of each label.

    Local copy of ``coloring_page.engines.region``'s private helper of the
    same name -- duplicated rather than imported since it isn't part of
    that module's public surface.
    """
    flat_lab = lab_image.reshape(-1, 3).astype(np.float64)
    flat_labels = labels.reshape(-1)
    sums = np.zeros((num_labels, 3), dtype=np.float64)
    counts = np.zeros(num_labels, dtype=np.float64)
    np.add.at(sums, flat_labels, flat_lab)
    np.add.at(counts, flat_labels, 1)
    return sums / np.maximum(counts, 1)[:, None]


def _adjacent_labels(
    contour: np.ndarray, labels: np.ndarray, height: int, width: int
) -> tuple[int, int] | None:
    """Find two distinct region labels bordering a traced boundary contour.

    Local copy of ``coloring_page.engines.region``'s private
    ``_labels_adjacent_to_contour`` -- duplicated rather than imported
    since it isn't part of that module's public surface.
    """
    seen: set[int] = set()
    for x, y in contour.reshape(-1, 2).tolist():
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                seen.add(int(labels[ny, nx]))
        if len(seen) >= 2:
            break
    if len(seen) < 2:
        return None
    first, second = sorted(seen)[:2]
    return first, second


def _merge_and_prune(
    labels: np.ndarray,
    detail_response: np.ndarray,
    image: np.ndarray,
    *,
    min_region_area_px2: float,
    min_region_contrast: float,
    detail_threshold: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Stage C: decide which region boundaries and detail lines are colorable.

    This is the stage that decides whether the output is colorable at
    all. A region boundary survives only if it separates two regions
    where at least one of the following holds: the smaller adjacent
    region's area clears ``min_region_area_px2``, or the two regions'
    mean Lab color distance clears ``min_region_contrast``. A region with
    *no* surviving boundary at all (every edge around it was too small
    and too low-contrast) is considered merged away entirely: any
    Stage B detail-line ink that falls inside it is dropped too, since a
    detail line drawn inside a region the child was never going to
    perceive as separate from its surroundings is noise, not signal --
    this is the mechanism that eliminates grass, paper grain, and
    background clutter without touching real subjects.

    Detail-line ink itself is kept only where the Stage B soft response
    clears ``detail_threshold`` -- applied via
    ``postprocess.hysteresis_centerline`` (never ``gradient_edges``, see
    :func:`_detail_lines`) -- and only within a kept region.

    Parameters
    ----------
    labels : np.ndarray
        Integer label image from :func:`_segment`, shape ``(H, W)``.
    detail_response : np.ndarray
        Soft response map from :func:`_detail_lines`, shape ``(H, W)``,
        ``uint8``, lower = more ink-like.
    image : np.ndarray
        BGR image, shape ``(H, W, 3)``, used to compute per-region mean
        Lab color.
    min_region_area_px2 : float
        Minimum area, in pixels squared, of the smaller of two regions a
        boundary separates, for that boundary to survive on area alone.
    min_region_contrast : float
        Minimum Lab Euclidean distance between two adjacent regions' mean
        colors, for their shared boundary to survive on contrast alone.
    detail_threshold : float
        ``postprocess.hysteresis_centerline``'s ``strong_threshold``,
        already on the 0-255 normalized-percentile scale
        :func:`_detail_lines` contrast-stretched ``detail_response`` to.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        Surviving boundary contours (each an ``Nx1x2`` int32 array, in
        ``(x, y)`` pixel order, as returned by ``cv2.findContours``), and
        the surviving detail-line mask (``uint8``, nonzero = ink, matching
        :func:`~coloring_page.drawing.paths_from_mask`'s contract).
    """
    height, width = labels.shape
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    num_labels = int(labels.max()) + 1
    mean_lab = _mean_lab_per_label(lab, labels, num_labels)
    label_areas = np.bincount(labels.reshape(-1), minlength=num_labels).astype(np.float64)

    kept_chains: list[np.ndarray] = []
    kept_labels: set[int] = set()
    for label_id in range(num_labels):
        mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            adjacency = _adjacent_labels(contour, labels, height, width)
            if adjacency is None:
                continue
            first, second = adjacency
            if label_id != first:
                # A boundary shared by two labels is found once per side
                # (once tracing each label's own contour); only keep it
                # from the lower-numbered side so it isn't emitted twice.
                continue
            smaller_area = min(label_areas[first], label_areas[second])
            contrast = float(np.linalg.norm(mean_lab[first] - mean_lab[second]))
            if smaller_area >= min_region_area_px2 or contrast >= min_region_contrast:
                kept_chains.append(contour)
                kept_labels.add(first)
                kept_labels.add(second)

    kept_region_mask = (
        np.isin(labels, list(kept_labels)) if kept_labels else np.zeros(labels.shape, dtype=bool)
    )

    weak_threshold = min(255.0, detail_threshold + 60.0)
    detail_binary = hysteresis_centerline(
        detail_response, strong_threshold=detail_threshold, weak_threshold=weak_threshold
    )
    detail_mask = np.where(kept_region_mask & (detail_binary > 0), 255, 0).astype(np.uint8)

    return kept_chains, detail_mask


def _chaikin_smooth(points: np.ndarray, *, iterations: int, closed: bool) -> np.ndarray:
    """Smooth a polyline via repeated Chaikin corner-cutting.

    Each iteration replaces every edge ``(P0, P1)`` with two new points at
    1/4 and 3/4 along it, rounding off every corner slightly; a handful of
    iterations turns an angular, ``approxPolyDP``-simplified polyline into
    the soft curve this engine's reference output (``docs/desired*.jpg``)
    has and the classical engines' jagged output doesn't. Chosen over
    cubic Bezier fitting: it operates directly on the already-simplified
    polyline with no fitting/optimization step, is trivial to reason about
    and unit-test (a square's corners get cut predictably), and is fully
    deterministic -- a fair match for this engine's otherwise
    deterministic pipeline.

    Parameters
    ----------
    points : np.ndarray
        Shape ``(N, 2)``.
    iterations : int
        Number of corner-cutting passes; 0 returns ``points`` unchanged.
    closed : bool
        Whether the last point connects back to the first. An open
        path's two endpoints are preserved exactly across every
        iteration; a closed path has no fixed endpoint to preserve.

    Returns
    -------
    np.ndarray
        Smoothed points, same dtype as ``points``.
    """
    result = points
    for _ in range(iterations):
        if len(result) < 3:
            break
        if closed:
            starts = result
            ends = np.roll(result, -1, axis=0)
        else:
            starts = result[:-1]
            ends = result[1:]
        q = 0.75 * starts + 0.25 * ends
        r = 0.25 * starts + 0.75 * ends
        cut = np.empty((len(q) * 2, 2), dtype=result.dtype)
        cut[0::2] = q
        cut[1::2] = r
        result = cut if closed else np.vstack([result[0], cut, result[-1]])
    return result


def _drop_hausdorff_duplicates(
    paths: tuple[DrawingPath, ...], *, threshold: float
) -> tuple[DrawingPath, ...]:
    """Drop a detail-kind path that duplicates a boundary-kind path tracing the same edge.

    A physical edge that both Stage A (a region boundary) and Stage B (the
    detail-line network) independently trace becomes two near-identical
    paths otherwise -- doubled ink this project's coloring pages never
    intend. Two paths are considered the same physical line when their
    Hausdorff distance (checked in both directions, since neither path is
    guaranteed to be the superset of the other) is below ``threshold``;
    when that happens the boundary-kind path is kept (Stage A's geometry
    comes from an exact label-map contour, more stable than Stage B's
    soft-response centerline) and the detail-kind path is dropped. Only
    boundary/detail pairs are compared -- same-kind boundary duplicates
    from a shared edge's two sides are already deduplicated in
    :func:`_merge_and_prune`.

    Parameters
    ----------
    paths : tuple[Path, ...]
        Candidate paths, normalized ``[0, 1]`` units (same units as
        ``threshold``).
    threshold : float
        Maximum directed Hausdorff distance, in the same normalized units
        as ``Path.points``, for two paths to be considered duplicates.

    Returns
    -------
    tuple[Path, ...]
        ``paths`` with duplicate detail-kind paths removed.
    """
    boundaries = [p for p in paths if p.kind == "boundary"]
    details = [p for p in paths if p.kind == "detail"]
    if not boundaries or not details:
        return paths

    kept_details = []
    for detail in details:
        is_duplicate = False
        for boundary in boundaries:
            forward = directed_hausdorff(detail.points, boundary.points)[0]
            backward = directed_hausdorff(boundary.points, detail.points)[0]
            if max(forward, backward) < threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_details.append(detail)

    return tuple(boundaries) + tuple(kept_details)


def _regularize(
    boundary_chains: list[np.ndarray],
    detail_mask: np.ndarray,
    image_shape: tuple[int, int],
    aspect_ratio: float,
    *,
    polyline_epsilon_px: float,
    min_path_length_px: float,
    hausdorff_threshold_px: float,
    chaikin_iterations: int,
) -> Drawing:
    """Stage D: vectorize, simplify, smooth, prune, and deduplicate into a final ``Drawing``.

    Parameters
    ----------
    boundary_chains : list[np.ndarray]
        Surviving boundary contours from :func:`_merge_and_prune`.
    detail_mask : np.ndarray
        Surviving detail-line mask from :func:`_merge_and_prune`.
    image_shape : tuple[int, int]
        ``(height, width)`` the above are relative to.
    aspect_ratio : float
        ``width / height``, stored on the returned :class:`Drawing`.
    polyline_epsilon_px : float
        ``cv2.approxPolyDP`` tolerance, in working-resolution pixels.
    min_path_length_px : float
        Paths shorter than this (in working-resolution pixels) are
        dropped.
    hausdorff_threshold_px : float
        Passed to :func:`_drop_hausdorff_duplicates`, in working-resolution
        pixels.
    chaikin_iterations : int
        Passed to :func:`_chaikin_smooth`.

    Returns
    -------
    Drawing
        The final vector line art.
    """
    long_side_px = max(image_shape)

    boundary_paths = paths_from_point_chains(
        boundary_chains, image_shape, closed=True, kind="boundary"
    )
    detail_paths = paths_from_mask(detail_mask, kind="detail") if detail_mask.any() else ()

    drawing = Drawing(paths=boundary_paths + detail_paths, aspect_ratio=aspect_ratio)
    if not drawing.paths:
        return drawing

    drawing = drawing.simplify(polyline_epsilon_px / long_side_px)

    smoothed_paths = tuple(
        dataclasses.replace(
            path,
            points=_chaikin_smooth(path.points, iterations=chaikin_iterations, closed=path.closed),
        )
        for path in drawing.paths
    )
    drawing = Drawing(paths=smoothed_paths, aspect_ratio=drawing.aspect_ratio, meta=drawing.meta)

    drawing = drawing.filter(lambda p: p.length * long_side_px >= min_path_length_px)

    deduplicated = _drop_hausdorff_duplicates(
        drawing.paths, threshold=hausdorff_threshold_px / long_side_px
    )
    return Drawing(paths=deduplicated, aspect_ratio=drawing.aspect_ratio, meta=drawing.meta)


def _labels_to_debug_image(labels: np.ndarray) -> np.ndarray:
    """Render a label map's region boundaries for debug inspection (0=ink, 255=background)."""
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return np.where(boundary, 0, 255).astype(np.uint8)


class LineArtEngine(ConversionEngine):
    """Segment into regions (SAM 2), add back fine detail, prune, regularize.

    Four independently-testable stages (:func:`_segment`,
    :func:`_detail_lines`, :func:`_merge_and_prune`, :func:`_regularize`) --
    see the module docstring for the overall rationale. Not yet measured
    against this project's reference images with real weights (see
    ``docs/PRODUCTION-PROMPTS.md``'s pre-merge gates); registered as
    ``experimental`` until that measurement happens.
    """

    name = "lineart"
    experimental = True
    requires_serial_execution = True

    def __init__(
        self,
        segmentation_resolution: int = 1024,
        detail_resolution: int = 1024,
        min_region_area_mm2: float = 4.0,
        min_region_contrast: float = 8.0,
        detail_threshold: float = 128.0,
        min_path_length_mm: float = 2.0,
        hausdorff_threshold_mm: float = 1.0,
        chaikin_iterations: int = 2,
        polyline_epsilon_px: float = 1.2,
        print_dpi: float = 300.0,
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        weights_path: str | Path | None = None,
        detail_weights_path: str | Path | None = None,
        mask_generator: Any | None = None,
        detail_model: Any | None = None,
    ) -> None:
        """Store this engine's stage parameters.

        Parameters
        ----------
        segmentation_resolution : int, optional
            Short-side working resolution :func:`_segment` runs SAM 2 at,
            by default 1024.
        detail_resolution : int, optional
            Short-side working resolution :func:`_detail_lines` runs the
            detail network at, by default 1024.
        min_region_area_mm2 : float, optional
            :func:`_merge_and_prune`'s area-survival threshold, in square
            millimeters at print size, by default 4.0.
        min_region_contrast : float, optional
            :func:`_merge_and_prune`'s Lab-contrast-survival threshold, by
            default 8.0. Not a physical unit (Lab distance is
            resolution-independent already), so not mm-based.
        detail_threshold : float, optional
            ``postprocess.hysteresis_centerline``'s ``strong_threshold``
            for Stage B's soft response, on the 0-255
            normalize-percentile scale, by default 128.0.
        min_path_length_mm : float, optional
            :func:`_regularize`'s minimum surviving path length, in
            millimeters at print size, by default 2.0.
        hausdorff_threshold_mm : float, optional
            :func:`_drop_hausdorff_duplicates`'s distance threshold, in
            millimeters at print size, by default 1.0.
        chaikin_iterations : int, optional
            Passed to :func:`_chaikin_smooth`, by default 2.
        polyline_epsilon_px : float, optional
            ``cv2.approxPolyDP`` tolerance, in working-resolution pixels
            (not mm -- this is a simplification tolerance relative to
            detection resolution, not a print-physical quantity), by
            default 1.2, matching other engines' ``polyline_epsilon``.
        print_dpi : float, optional
            Resolution, in dots per inch, used to convert this engine's
            millimeter-based parameters to working-resolution pixels, by
            default 300.0.
        device : {"auto", "cpu", "cuda", "mps"}, optional
            Inference device for both models, by default ``"auto"``
            (picks CUDA, then MPS, then CPU -- see :func:`_resolve_device`).
        weights_path, detail_weights_path : str | Path | None, optional
            Explicit checkpoint paths, overriding the default lookup (env
            var, then ``weights/``, then the per-user cache directory) --
            see :func:`_resolve_weights_path`.
        mask_generator, detail_model : Any | None, optional
            Pre-built model objects, injected in place of lazily building
            them from weights on first use. This is what lets tests
            substitute a stub without any real weights or GPU.
        """
        self.segmentation_resolution = segmentation_resolution
        self.detail_resolution = detail_resolution
        self.min_region_area_mm2 = min_region_area_mm2
        self.min_region_contrast = min_region_contrast
        self.detail_threshold = detail_threshold
        self.min_path_length_mm = min_path_length_mm
        self.hausdorff_threshold_mm = hausdorff_threshold_mm
        self.chaikin_iterations = chaikin_iterations
        self.polyline_epsilon_px = polyline_epsilon_px
        self.print_dpi = print_dpi
        self.device = device
        self.weights_path = weights_path
        self.detail_weights_path = detail_weights_path
        self._mask_generator = mask_generator
        self._detail_model = detail_model

    def convert(self, image: np.ndarray, *, debug: DebugSink | None = None) -> Drawing:
        """Run all four stages and return the final ``Drawing``.

        See Also
        --------
        ConversionEngine.convert : Full parameter and return-value contract.
        """
        height, width = image.shape[:2]
        long_side_px = max(height, width)
        resolved_device = _resolve_device(self.device)

        mask_generator = self._mask_generator
        if mask_generator is None:
            seg_weights = _resolve_lineart_weights_path(
                self.weights_path,
                "sam2.1_hiera_small.pt",
                WEIGHTS_ENV_VAR,
                _LOCAL_SEGMENTATION_WEIGHTS_PATH,
            )
            mask_generator = _build_mask_generator(seg_weights, device=resolved_device)

        detail_model = self._detail_model
        if detail_model is None:
            detail_weights = _resolve_lineart_weights_path(
                self.detail_weights_path,
                "netG.pth",
                DETAIL_WEIGHTS_ENV_VAR,
                _LOCAL_DETAIL_WEIGHTS_PATH,
            )
            detail_model = _build_detail_model(detail_weights, device=resolved_device)

        labels = _segment(image, mask_generator, resolution=self.segmentation_resolution)
        if debug is not None:
            debug.save("segments", _labels_to_debug_image(labels))

        detail_response = _detail_lines(image, detail_model, resolution=self.detail_resolution)
        if debug is not None:
            debug.save("detail_response", detail_response)

        min_region_area_px2 = _mm2_to_px2(self.min_region_area_mm2, dpi=self.print_dpi)
        boundary_chains, detail_mask = _merge_and_prune(
            labels,
            detail_response,
            image,
            min_region_area_px2=min_region_area_px2,
            min_region_contrast=self.min_region_contrast,
            detail_threshold=self.detail_threshold,
        )
        if debug is not None:
            debug.save("detail_mask", np.where(detail_mask > 0, 0, 255).astype(np.uint8))

        drawing = _regularize(
            boundary_chains,
            detail_mask,
            (height, width),
            width / height,
            polyline_epsilon_px=self.polyline_epsilon_px,
            min_path_length_px=_mm_to_px(self.min_path_length_mm, dpi=self.print_dpi),
            hausdorff_threshold_px=_mm_to_px(self.hausdorff_threshold_mm, dpi=self.print_dpi),
            chaikin_iterations=self.chaikin_iterations,
        )
        if debug is not None:
            debug.save("redraw", rasterize(drawing, long_side_px=long_side_px))
        return drawing
