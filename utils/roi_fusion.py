"""Consistent fusion of per-crop depth predictions into a global prediction.

Implements Step 1 of docs/pre_depth_improvement_plan.md: each crop prediction
is scale-shift aligned to the global prediction (least squares) before being
pasted, so the composite "pre-depth" stays globally consistent.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from utils.depth_alignment import apply_scale_shift, fit_scale_shift
from utils.semantic_mask_utils import make_soft_weight

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def resize_depth(depth: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    return np.array(
        Image.fromarray(depth.astype(np.float32), mode="F").resize(size_wh, resample=Image.BILINEAR)
    )


def fuse_roi_depth(
    global_depth: np.ndarray,
    crop_preds: Sequence[Tuple[Box, np.ndarray]],
    roi_mask: Optional[np.ndarray],
    align_mode: str = "lstsq",
    blend_blur_ksize: int = 31,
    fusion_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse crop predictions onto the global prediction.

    Args:
        global_depth: [H, W] global prediction (depth or disparity).
        crop_preds: list of ((x1, y1, x2, y2), crop_prediction). Crop
            predictions may have any resolution; they are resized to the box.
        roi_mask: [H, W] object mask in [0, 1] used for soft blending.
            If None, the pasted box areas are used directly.
        align_mode: "lstsq" fits each crop to the global prediction inside its
            box before pasting; "none" pastes raw crop values (previous
            behaviour, kept for ablation).
        blend_blur_ksize: Gaussian kernel for the soft blend weight.
        fusion_weight: multiplier on the blend weight (1.0 = full fusion).

    Returns:
        (fused_depth, effective_mask): fused map and the mask of pixels that
        received crop information (before soft blurring).
    """
    if align_mode not in ("lstsq", "none"):
        raise ValueError(f"Unknown align_mode: {align_mode}")

    h, w = global_depth.shape
    acc = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)
    effective_mask = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2), crop_depth in crop_preds:
        if x2 <= x1 or y2 <= y1:
            continue
        crop_resized = resize_depth(crop_depth, (x2 - x1, y2 - y1))
        if align_mode == "lstsq":
            scale, shift = fit_scale_shift(crop_resized, global_depth[y1:y2, x1:x2])
            crop_resized = apply_scale_shift(crop_resized, scale, shift)
        # Overlapping boxes are averaged instead of overwritten.
        acc[y1:y2, x1:x2] += crop_resized
        weight[y1:y2, x1:x2] += 1.0
        if roi_mask is not None:
            effective_mask[y1:y2, x1:x2] = np.maximum(
                effective_mask[y1:y2, x1:x2], roi_mask[y1:y2, x1:x2]
            )
        else:
            effective_mask[y1:y2, x1:x2] = 1.0

    pasted = weight > 0
    canvas = global_depth.astype(np.float64).copy()
    canvas[pasted] = acc[pasted] / weight[pasted]

    soft = make_soft_weight(effective_mask, blur_ksize=blend_blur_ksize)
    wmap = np.clip(soft * fusion_weight, 0.0, 1.0)
    # Only blend where crop information exists.
    wmap = wmap * pasted.astype(np.float32)
    fused = wmap * canvas + (1.0 - wmap) * global_depth
    return fused.astype(np.float32), effective_mask
