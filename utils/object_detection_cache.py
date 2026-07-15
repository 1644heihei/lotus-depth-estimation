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
        if masks is not None and i < len(masks):
            seg = masks[i].detach().cpu().numpy()
            if seg.ndim == 3:
                seg = seg[0]
            seg_resized = cv2.resize(seg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            mask_area = float((seg_resized > 0.5).sum())
        detections.append(
            ObjectDetection(
                bbox=[x1, y1, x2, y2],
                class_id=cls_id,
                class_name=label,
                score=score,
                mask_area=mask_area,
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
