"""YOLO bbox detection for object crop pipeline (no segmentation masks)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

Box = Tuple[int, int, int, int]

DEFAULT_INDOOR_LABELS = {
    "chair", "couch", "bed", "dining table", "tv", "laptop", "book", "vase",
    "potted plant", "toilet", "sink", "refrigerator", "microwave", "oven",
    "clock", "cup", "bottle", "bowl", "remote", "keyboard", "cell phone",
    "teddy bear", "backpack", "handbag", "suitcase", "bench",
}


def bboxes_to_roi_mask(h: int, w: int, boxes: Sequence[Box]) -> np.ndarray:
    """Fill axis-aligned boxes into a float ROI map [H, W] in {0, 1}."""
    mask = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    return mask


def _box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def parse_label_filter(labels_arg: str) -> Optional[Set[str]]:
    if not labels_arg.strip():
        return None
    return {x.strip().lower() for x in labels_arg.split(",") if x.strip()}


class YoloBboxDetector:
    """Ultralytics YOLO detect (bbox only, not seg)."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        score_thr: float = 0.25,
        labels_keep: Optional[Set[str]] = None,
        device: str = "0",
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.score_thr = score_thr
        self.labels_keep = labels_keep
        self.device = device
        self.imgsz = imgsz

    def detect(
        self,
        rgb_np: np.ndarray,
        *,
        min_area: int = 500,
    ) -> List[Box]:
        h, w = rgb_np.shape[:2]
        results = self.model.predict(
            source=rgb_np,
            imgsz=self.imgsz,
            conf=self.score_thr,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes: List[Box] = []
        if result.boxes is None:
            return boxes

        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = str(names.get(cls_id, "")).lower()
            if self.labels_keep is not None and label not in self.labels_keep:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            b: Box = (x1, y1, x2, y2)
            if _box_area(b) >= min_area:
                boxes.append(b)
        return boxes
