"""Core-model ROI fusion to build pre-depth maps (Approach A, Step 2)."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from utils.object_detection_cache import ObjectDetection, detections_to_mask
from utils.roi_fusion import fuse_roi_depth
from utils.semantic_mask_utils import connected_component_boxes, expand_box


def pre_depth_path(rgb_path: str | Path, detail_root: str | Path) -> Path:
    from utils.object_detection_cache import _stem_and_parent

    detail_root = Path(detail_root)
    parent, stem = _stem_and_parent(rgb_path)
    return detail_root / parent / f"{stem}_pre_depth.npy"


def valid_mask_path(rgb_path: str | Path, detail_root: str | Path) -> Path:
    from utils.object_detection_cache import _stem_and_parent

    detail_root = Path(detail_root)
    parent, stem = _stem_and_parent(rgb_path)
    return detail_root / parent / f"{stem}_valid_mask.npy"


def disparity_pred_to_norm(pred_01: np.ndarray) -> np.ndarray:
    """Map Lotus-D disparity output in [0, 1] to [-1, 1]."""
    return np.clip(pred_01.astype(np.float32), 0.0, 1.0) * 2.0 - 1.0


class CoreDepthPredictor:
    """Wrap Lotus-D pipeline for global + per-bbox depth prediction."""

    def __init__(
        self,
        pipe,
        *,
        timestep: int = 999,
        processing_res: Optional[int] = None,
        resample_method: str = "bilinear",
        generator: Optional[torch.Generator] = None,
    ):
        self.pipe = pipe
        self.timestep = timestep
        self.processing_res = processing_res
        self.resample_method = resample_method
        self.generator = generator

    @torch.no_grad()
    def predict_rgb(self, rgb_np: np.ndarray) -> np.ndarray:
        device = self.pipe.device
        image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        image = image / 127.5 - 1.0
        image = image.to(device)
        task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(device_type=device.type)
        with autocast_ctx:
            pred = self.pipe(
                rgb_in=image,
                prompt="",
                num_inference_steps=1,
                generator=self.generator,
                output_type="np",
                timesteps=[self.timestep],
                task_emb=task_emb,
                processing_res=self.processing_res,
                match_input_res=True,
                resample_method=self.resample_method,
            ).images[0]
        return pred.mean(axis=-1).astype(np.float32)

    def build_pre_depth(
        self,
        rgb_np: np.ndarray,
        detections: Sequence[ObjectDetection],
        *,
        roi_min_area: int = 500,
        roi_expand_ratio: float = 0.25,
        align_mode: str = "lstsq",
        blend_blur_ksize: int = 31,
        fusion_weight: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pre_depth_norm, valid_mask, global_depth_01)."""
        h, w = rgb_np.shape[:2]
        global_depth = self.predict_rgb(rgb_np)
        roi_mask = detections_to_mask(detections, h, w)

        boxes = connected_component_boxes(roi_mask, threshold=0.5, min_area=roi_min_area)
        if not boxes and detections:
            boxes = [tuple(det.bbox) for det in detections]

        crop_preds: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
        for box in boxes:
            x1, y1, x2, y2 = expand_box(box, (h, w), expand_ratio=roi_expand_ratio)
            crop = rgb_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_depth = self.predict_rgb(crop)
            crop_preds.append(((x1, y1, x2, y2), crop_depth))

        fused_depth, effective_mask = fuse_roi_depth(
            global_depth=global_depth,
            crop_preds=crop_preds,
            roi_mask=roi_mask,
            align_mode=align_mode,
            blend_blur_ksize=blend_blur_ksize,
            fusion_weight=fusion_weight,
        )
        pre_depth_norm = disparity_pred_to_norm(fused_depth)
        return pre_depth_norm, effective_mask.astype(np.float32), global_depth


def save_pre_depth_artifacts(
    pre_depth_norm: np.ndarray,
    valid_mask: np.ndarray,
    rgb_path: str | Path,
    detail_root: str | Path,
) -> Tuple[Path, Path]:
    pre_path = pre_depth_path(rgb_path, detail_root)
    valid_path = valid_mask_path(rgb_path, detail_root)
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pre_path, pre_depth_norm.astype(np.float32))
    np.save(valid_path, valid_mask.astype(np.float32))
    return pre_path, valid_path


def load_pre_depth_artifacts(rgb_path: str | Path, detail_root: str | Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    pre_p = pre_depth_path(rgb_path, detail_root)
    valid_p = valid_mask_path(rgb_path, detail_root)
    pre = np.load(pre_p) if pre_p.is_file() else None
    valid = np.load(valid_p) if valid_p.is_file() else None
    return pre, valid
