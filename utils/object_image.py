"""Build object-only RGB crops for per-object depth estimation (Phase 1 Step 1)."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

ObjectImageMode = Literal["crop", "masked"]


def make_object_image(
    rgb: np.ndarray,
    roi_mask: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    mode: ObjectImageMode = "crop",
    fill_value: int = 0,
) -> np.ndarray:
    """Extract an object-focused RGB patch from a full image.

    Args:
        rgb: uint8 RGB [H, W, 3].
        roi_mask: float mask [H, W] in [0, 1].
        box: (x1, y1, x2, y2) in full-image coordinates (already expanded).
        mode:
            crop   - plain bbox crop (background kept; default, Lotus-friendly).
            masked - zero out pixels outside the object mask inside the box.
        fill_value: background fill for masked mode (0 = black).

    Returns:
        uint8 RGB crop [h, w, 3].
    """
    x1, y1, x2, y2 = box
    crop = rgb[y1:y2, x1:x2].copy()
    if mode == "crop":
        return crop

    if mode != "masked":
        raise ValueError(f"Unknown object image mode: {mode}")

    local_mask = roi_mask[y1:y2, x1:x2]
    keep = (local_mask > 0.5)[..., None]
    background = np.full_like(crop, fill_value, dtype=np.uint8)
    return np.where(keep, crop, background)
