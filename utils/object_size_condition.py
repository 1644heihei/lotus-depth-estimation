"""BBox size maps co-rasterized with class_map (Approach A)."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from utils.object_detection_cache import ObjectDetection


def size_map_to_tensor(size_map: np.ndarray) -> np.ndarray:
    """Map [0, 1] size values to [-1, 1] for UNet conditioning."""
    sm = np.clip(size_map.astype(np.float32), 0.0, 1.0)
    return sm * 2.0 - 1.0


def rasterize_class_and_size_maps(
    detections: Sequence[ObjectDetection],
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paint class_id + normalized bbox w/h with a shared score-priority winner.

    Returns:
        class_map: uint16 HxW (0=bg, else class_id+1)
        size_w: float32 HxW in [0, 1] = (x2-x1)/width
        size_h: float32 HxW in [0, 1] = (y2-y1)/height
    """
    class_map = np.zeros((height, width), dtype=np.uint16)
    size_w = np.zeros((height, width), dtype=np.float32)
    size_h = np.zeros((height, width), dtype=np.float32)
    score_map = np.full((height, width), -1.0, dtype=np.float32)

    img_w = max(int(width), 1)
    img_h = max(int(height), 1)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, min(int(x1), img_w))
        x2 = max(0, min(int(x2), img_w))
        y1 = max(0, min(int(y1), img_h))
        y2 = max(0, min(int(y2), img_h))
        if x2 <= x1 or y2 <= y1:
            continue

        region_score = score_map[y1:y2, x1:x2]
        update = det.score >= region_score
        if not update.any():
            continue

        w_norm = float(x2 - x1) / float(img_w)
        h_norm = float(y2 - y1) / float(img_h)

        class_patch = class_map[y1:y2, x1:x2]
        class_patch[update] = np.uint16(det.class_id + 1)

        sw = size_w[y1:y2, x1:x2]
        sh = size_h[y1:y2, x1:x2]
        sw[update] = w_norm
        sh[update] = h_norm

        score_map[y1:y2, x1:x2] = np.where(update, det.score, region_score)

    return class_map, size_w, size_h
