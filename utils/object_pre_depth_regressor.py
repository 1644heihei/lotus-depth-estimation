"""Build pre-depth maps from object depth regressor predictions."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from utils.object_detection_cache import ObjectDetection
from utils.object_depth_regressor import ObjectDepthRegressorBundle
from utils.object_pre_depth import disparity_pred_to_norm
from utils.semantic_mask_utils import make_soft_weight


def _central_bbox_median(
    disparity: np.ndarray, bbox: Sequence[int], central_ratio: float = 0.5
) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = disparity.shape[:2]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    dx = int((x2 - x1) * (1.0 - central_ratio) * 0.5)
    dy = int((y2 - y1) * (1.0 - central_ratio) * 0.5)
    patch = disparity[y1 + dy : y2 - dy, x1 + dx : x2 - dx]
    valid = np.isfinite(patch)
    return float(np.median(patch[valid])) if valid.any() else float("nan")


def fit_relative_disparity_calibration(
    metric_disparities: np.ndarray,
    global_disparities: np.ndarray,
) -> Tuple[float, float]:
    """Fit positive-slope y=a*x+b, with shift-only fallback for degenerate input."""
    x = np.asarray(metric_disparities, dtype=np.float64)
    y = np.asarray(global_disparities, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size == 0:
        return 1.0, 0.0
    shift = float(np.median(y - x))
    if x.size < 2 or float(np.var(x)) < 1e-12 or float(np.var(y)) < 1e-12:
        return 1.0, shift
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = np.abs(scale * x + bias - y)
    if x.size >= 4:
        keep = residual <= np.quantile(residual, 0.75)
        if keep.sum() >= 2 and float(np.var(x[keep])) >= 1e-12:
            design = np.stack([x[keep], np.ones_like(x[keep])], axis=1)
            scale, bias = np.linalg.lstsq(design, y[keep], rcond=None)[0]
    if not np.isfinite(scale) or scale <= 1e-8 or not np.isfinite(bias):
        return 1.0, shift
    return float(scale), float(bias)


def build_pre_depth_from_regressor(
    global_depth_01: np.ndarray,
    detections: Sequence[ObjectDetection],
    regressor: ObjectDepthRegressorBundle,
    *,
    blend_blur_ksize: int = 31,
    fusion_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse regressor constant-depth boxes onto a global Lotus disparity map.

    Args:
        global_depth_01: [H,W] disparity in [0,1] from Lotus-D global pass.
        detections: filtered YOLO detections.
        regressor: trained object depth bundle.

    Returns:
        (pre_depth_norm [-1,1], valid_mask [0,1])
    """
    h, w = global_depth_01.shape[:2]
    img_w = max(int(w), 1)
    img_h = max(int(h), 1)
    roi_mask = np.zeros((h, w), dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        roi_mask[y1:y2, x1:x2] = 1.0
    valid_detections = []
    metric_disparities = []
    global_disparities = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if x2 <= x1 or y2 <= y1:
            continue
        bw = float(x2 - x1) / float(img_w)
        bh = float(y2 - y1) / float(img_h)
        cy = float(y1 + y2) * 0.5 / float(img_h)
        depth_m = regressor.predict_depth_m(
            det.class_id,
            bw,
            bh,
            cy,
            score=float(det.score),
            mask_fill_ratio=float(det.mask_fill_ratio),
            mask_axis_ratio=float(det.mask_axis_ratio),
            mask_angle_sin2=float(det.mask_angle_sin2),
            mask_angle_cos2=float(det.mask_angle_cos2),
            mask_orientation_confidence=float(det.mask_orientation_confidence),
            mask_valid=float(det.mask_valid),
        )
        metric_disp = 1.0 / max(depth_m, 1e-4)
        global_disp = _central_bbox_median(global_depth_01, det.bbox)
        if not np.isfinite(global_disp):
            continue
        valid_detections.append(det)
        metric_disparities.append(metric_disp)
        global_disparities.append(global_disp)

    if not valid_detections:
        return disparity_pred_to_norm(global_depth_01), np.zeros((h, w), dtype=np.float32)

    scale, shift = fit_relative_disparity_calibration(
        np.asarray(metric_disparities), np.asarray(global_disparities)
    )
    calibrated = np.clip(scale * np.asarray(metric_disparities) + shift, 0.0, 1.0)

    canvas = global_depth_01.astype(np.float32).copy()
    score_map = np.full((h, w), -1.0, dtype=np.float32)
    effective_mask = np.zeros((h, w), dtype=np.float32)
    for det, value in zip(valid_detections, calibrated):
        x1, y1, x2, y2 = det.bbox
        region_score = score_map[y1:y2, x1:x2]
        update = det.score >= region_score
        patch = canvas[y1:y2, x1:x2]
        patch[update] = np.float32(value)
        region_mask = effective_mask[y1:y2, x1:x2]
        region_mask[update] = roi_mask[y1:y2, x1:x2][update]
        score_map[y1:y2, x1:x2] = np.where(update, det.score, region_score)

    pasted = score_map >= 0
    soft = make_soft_weight(effective_mask, blur_ksize=blend_blur_ksize)
    wmap = np.clip(soft * fusion_weight, 0.0, 1.0) * pasted.astype(np.float32)
    fused = wmap * canvas + (1.0 - wmap) * global_depth_01
    pre_depth_norm = disparity_pred_to_norm(fused)
    return pre_depth_norm, effective_mask.astype(np.float32)


def build_pre_depth_for_rgb(
    rgb_np: np.ndarray,
    detections: Sequence[ObjectDetection],
    regressor: ObjectDepthRegressorBundle,
    core_predictor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run global Lotus + regressor fusion.

    Returns:
        pre_depth_norm, valid_mask, global_depth_01
    """
    global_depth = core_predictor.predict_rgb(rgb_np)
    pre_depth_norm, valid_mask = build_pre_depth_from_regressor(
        global_depth,
        detections,
        regressor,
    )
    return pre_depth_norm, valid_mask, global_depth
