"""Depth boundary F-score.

Edges are extracted from robustly normalized depth maps with Canny, then
matched with a pixel-distance tolerance (precision on pred edges, recall on
GT edges). This rewards exactly what pre-depth fusion is supposed to improve:
sharp, correctly located object boundaries.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def _normalize_robust(depth: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    d = depth.astype(np.float32)
    vals = d[valid_mask] if valid_mask is not None else d.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(d)
    lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
    if hi - lo < 1e-8:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


def depth_edges(
    depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    canny_low: int = 30,
    canny_high: int = 90,
) -> np.ndarray:
    """Binary edge map of a depth map (uint8 0/1)."""
    norm = _normalize_robust(depth, valid_mask)
    img = (norm * 255).astype(np.uint8)
    edges = cv2.Canny(img, canny_low, canny_high) > 0
    if valid_mask is not None:
        edges &= valid_mask.astype(bool)
    return edges.astype(np.uint8)


def boundary_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    tolerance_px: int = 3,
    canny_low: int = 30,
    canny_high: int = 90,
) -> Tuple[float, float, float]:
    """Boundary (precision, recall, F1) between pred and GT depth edges.

    A predicted edge pixel counts as correct if a GT edge exists within
    ``tolerance_px`` pixels, and vice versa for recall.
    """
    pred_e = depth_edges(pred, valid_mask, canny_low, canny_high)
    gt_e = depth_edges(gt, valid_mask, canny_low, canny_high)

    if gt_e.sum() == 0 and pred_e.sum() == 0:
        return 1.0, 1.0, 1.0
    if gt_e.sum() == 0 or pred_e.sum() == 0:
        return 0.0, 0.0, 0.0

    # Distance from every pixel to the nearest edge pixel.
    dist_to_gt = cv2.distanceTransform(1 - gt_e, cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform(1 - pred_e, cv2.DIST_L2, 3)

    precision = float((dist_to_gt[pred_e > 0] <= tolerance_px).mean())
    recall = float((dist_to_pred[gt_e > 0] <= tolerance_px).mean())
    if precision + recall < 1e-8:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
