"""Class-map generation from YOLO detections (Approach A, Step 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from utils.object_detection_cache import ObjectDetection, detections_json_path, load_detections


def class_map_path(rgb_path: str | Path, detail_root: str | Path) -> Path:
    from utils.object_detection_cache import _stem_and_parent

    detail_root = Path(detail_root)
    parent, stem = _stem_and_parent(rgb_path)
    return detail_root / parent / f"{stem}_class_map.npy"


def rasterize_class_map(
    detections: Sequence[ObjectDetection],
    height: int,
    width: int,
) -> np.ndarray:
    """Paint class_id per pixel; higher score wins on overlap."""
    class_map = np.zeros((height, width), dtype=np.uint16)
    score_map = np.full((height, width), -1.0, dtype=np.float32)
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        region_score = score_map[y1:y2, x1:x2]
        update = det.score >= region_score
        if not update.any():
            continue
        patch = class_map[y1:y2, x1:x2]
        patch[update] = np.uint16(det.class_id + 1)  # 0 reserved for background
        score_map[y1:y2, x1:x2] = np.where(update, det.score, region_score)
    return class_map


def save_class_map(class_map: np.ndarray, rgb_path: str | Path, detail_root: str | Path) -> Path:
    out_path = class_map_path(rgb_path, detail_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, class_map)
    return out_path


def load_class_map(rgb_path: str | Path, detail_root: str | Path) -> np.ndarray | None:
    path = class_map_path(rgb_path, detail_root)
    if not path.is_file():
        return None
    return np.load(path)


def class_map_to_tensor(class_map: np.ndarray, max_class_id: int = 90) -> np.ndarray:
    """Normalize class ids to [-1, 1] for VAE encoding."""
    cm = class_map.astype(np.float32)
    cm = np.clip(cm / max(max_class_id, 1), 0.0, 1.0)
    return cm * 2.0 - 1.0


def build_class_map_for_image(rgb_path: str | Path, detail_root: str | Path, height: int, width: int) -> np.ndarray:
    detections = load_detections(rgb_path, detail_root)
    return rasterize_class_map(detections, height, width)
