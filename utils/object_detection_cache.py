"""YOLO detection cache for Approach-A offline detail training dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import cv2
import numpy as np
from PIL import Image


@dataclass
class ObjectDetection:
    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    score: float
    mask_area: float = 0.0
    mask_fill_ratio: float = 1.0
    mask_axis_ratio: float = 1.0
    mask_angle_sin2: float = 0.0
    mask_angle_cos2: float = 1.0
    mask_orientation_confidence: float = 0.0
    mask_valid: float = 0.0


def mask_shape_features(binary_mask: np.ndarray, bbox_area: float) -> Dict[str, float]:
    """Return rotation-invariant and 180-degree-periodic 2D mask descriptors."""
    ys, xs = np.nonzero(binary_mask)
    fill_ratio = float(len(xs) / max(bbox_area, 1.0))
    if len(xs) < 4:
        return {
            "mask_fill_ratio": fill_ratio,
            "mask_axis_ratio": 1.0,
            "mask_angle_sin2": 0.0,
            "mask_angle_cos2": 1.0,
            "mask_orientation_confidence": 0.0,
            "mask_valid": 0.0,
        }
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = max(float(eigenvalues[order[0]]), 1e-8)
    minor = max(float(eigenvalues[order[1]]), 1e-8)
    direction = eigenvectors[:, order[0]]
    theta = float(np.arctan2(direction[1], direction[0]))
    axis_ratio = float(np.sqrt(major / minor))
    confidence = float(np.clip((major - minor) / (major + minor), 0.0, 1.0))
    return {
        "mask_fill_ratio": float(np.clip(fill_ratio, 0.0, 1.0)),
        "mask_axis_ratio": float(np.clip(axis_ratio, 1.0, 20.0)),
        "mask_angle_sin2": float(np.sin(2.0 * theta)),
        "mask_angle_cos2": float(np.cos(2.0 * theta)),
        "mask_orientation_confidence": confidence,
        "mask_valid": 1.0,
    }


def _stem_and_parent(rgb_path: str | Path) -> tuple[Path, str]:
    s = str(rgb_path)
    if s.startswith("hf://"):
        rel = _relative_under_split(s)
        return rel.parent, rel.stem
    p = Path(s)
    rel = _relative_under_split(p)
    return rel.parent, p.stem


def detections_json_path(rgb_path: str | Path, detail_root: str | Path) -> Path:
    detail_root = Path(detail_root)
    parent, stem = _stem_and_parent(rgb_path)
    return detail_root / parent / f"{stem}_detections.json"


def _relative_under_split(rgb_path: Path | str) -> Path:
    s = str(rgb_path)
    if s.startswith("hf://"):
        without = s[len("hf://") :]
        if "?" in without:
            without = without.split("?", 1)[0]
        parts = without.split("/")
        if len(parts) >= 3:
            return Path(*parts[2:])
        return Path(without)
    parts = Path(s).parts
    for idx, part in enumerate(parts):
        if part.lower().startswith("scene") and idx + 1 < len(parts):
            return Path(*parts[idx:])
    for split in ("train", "val", "test"):
        if split in parts:
            idx = parts.index(split)
            return Path(*parts[idx:])
    p = Path(s)
    return Path(p.parent.name) / p.name


def save_detections(rgb_path: str | Path, detail_root: str | Path, detections: Sequence[ObjectDetection]) -> Path:
    out_path = detections_json_path(rgb_path, detail_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(rgb_path),
        "detections": [asdict(d) for d in detections],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_detections(rgb_path: str | Path, detail_root: str | Path) -> List[ObjectDetection]:
    path = detections_json_path(rgb_path, detail_root)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ObjectDetection(**item) for item in payload.get("detections", [])]


def run_yolo_detections(
    rgb_np: np.ndarray,
    *,
    model,
    score_thr: float = 0.25,
    labels_keep: Optional[Set[str]] = None,
    imgsz: int = 640,
) -> List[ObjectDetection]:
    h, w = rgb_np.shape[:2]
    results = model.predict(source=rgb_np, imgsz=imgsz, conf=score_thr, verbose=False)
    result = results[0]
    detections: List[ObjectDetection] = []
    if result.boxes is None:
        return detections

    names: Dict[int, str] = result.names
    masks = result.masks.data if result.masks is not None else None
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item())
        score = float(box.conf.item())
        label = str(names.get(cls_id, "")).lower()
        if labels_keep is not None and label not in labels_keep:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        mask_area = float((x2 - x1) * (y2 - y1))
        shape_features = mask_shape_features(
            np.ones((y2 - y1, x2 - x1), dtype=bool), mask_area
        )
        shape_features["mask_valid"] = 0.0
        if masks is not None and i < len(masks):
            seg = masks[i].detach().cpu().numpy()
            if seg.ndim == 3:
                seg = seg[0]
            seg_resized = cv2.resize(seg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            binary_mask = seg_resized > 0.5
            mask_area = float(binary_mask.sum())
            shape_features = mask_shape_features(
                binary_mask[y1:y2, x1:x2],
                float((x2 - x1) * (y2 - y1)),
            )
        detections.append(
            ObjectDetection(
                bbox=[x1, y1, x2, y2],
                class_id=cls_id,
                class_name=label,
                score=score,
                mask_area=mask_area,
                **shape_features,
            )
        )
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def detections_to_mask(
    detections: Sequence[ObjectDetection],
    height: int,
    width: int,
    *,
    use_boxes: bool = True,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if use_boxes:
            mask[y1:y2, x1:x2] = 1.0
    return np.clip(mask, 0.0, 1.0)


def load_yolo_model(model_path: str = "yolov8n-seg.pt"):
    from ultralytics import YOLO

    return YOLO(model_path)
