"""SAM 2 automatic-mask segmentation, promoted from ``scripts/spike_segmentation.py``.

Only the label-map construction the spike validated is kept here --
``merge_small_regions``/``draw_label_boundaries``/``count_dangling_endpoints``
stay in the spike script, since those are QA tools for the PASS/FAIL
decision, not part of the production pipeline: ``lineart.py``'s own Stage C
(``_merge_and_prune``) does the equivalent small-region handling with
physical (mm-based) thresholds instead of the spike's fixed area fraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

#: Sentinel for a not-yet-assigned pixel in the label map under construction.
_UNASSIGNED = -1


def load_mask_generator(
    checkpoint: Path,
    *,
    device: str,
    config_file: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
) -> Any:
    """Build a SAM 2.1 Hiera-small automatic mask generator.

    Parameters
    ----------
    checkpoint : Path
        Path to the ``sam2.1_hiera_small.pt`` checkpoint, as fetched by
        ``scripts/fetch_weights.py``.
    device : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``, passed straight to
        ``sam2.build_sam.build_sam2``.
    config_file : str, optional
        SAM 2 model config, by default the Hiera-small config matching
        ``sam2.1_hiera_small.pt``.

    Returns
    -------
    Any
        A ``sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator``,
        typed ``Any`` since ``sam2`` is an optional dependency not present
        in this project's default type-checking environment.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(config_file=config_file, ckpt_path=str(checkpoint), device=device)
    return SAM2AutomaticMaskGenerator(model)


def generate_masks(mask_generator: Any, image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """Run SAM 2's automatic mask generator over a BGR image.

    Parameters
    ----------
    mask_generator : Any
        As returned by :func:`load_mask_generator`, or any object exposing
        a compatible ``.generate(image_rgb) -> list[dict]`` method (tests
        inject a stub here instead of a real SAM 2 model).
    image_bgr : np.ndarray
        BGR image, shape ``(H, W, 3)``, dtype ``uint8``.

    Returns
    -------
    list[dict[str, Any]]
        One dict per proposed mask, each with (among other keys) a boolean
        ``"segmentation"`` array of shape ``(H, W)`` and a numeric ``"area"``.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result: list[dict[str, Any]] = mask_generator.generate(image_rgb)
    return result


def build_label_map(masks: list[dict[str, Any]], image_shape: tuple[int, int]) -> np.ndarray:
    """Resolve overlapping SAM masks into one complete, non-overlapping label map.

    Each pixel is assigned to the *smallest* mask that contains it: masks
    are painted smallest-area-first, so a later (larger) mask only fills
    pixels no smaller mask already claimed. Pixels no mask covers at all
    are filled from their nearest already-assigned pixel (spatial nearest-
    neighbor via a distance transform).

    Parameters
    ----------
    masks : list[dict[str, Any]]
        As returned by :func:`generate_masks`.
    image_shape : tuple[int, int]
        ``(height, width)`` of the source image.

    Returns
    -------
    np.ndarray
        Integer label image, shape ``image_shape``, dtype ``int32``, with
        every pixel assigned to some label and no holes or overlaps.

    Raises
    ------
    ValueError
        If ``masks`` is empty or covers none of the image.
    """
    labels = np.full(image_shape, _UNASSIGNED, dtype=np.int32)
    for new_id, mask in enumerate(sorted(masks, key=lambda m: m["area"])):
        segmentation = mask["segmentation"]
        labels[segmentation & (labels == _UNASSIGNED)] = new_id

    unassigned = labels == _UNASSIGNED
    if unassigned.any():
        if unassigned.all():
            raise ValueError("SAM produced no masks covering any pixel of this image.")
        _, indices = ndimage.distance_transform_edt(unassigned, return_indices=True)
        nearest_labels = labels[tuple(indices)]
        labels[unassigned] = nearest_labels[unassigned]

    return labels
