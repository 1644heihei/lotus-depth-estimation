"""Generate semantic masks for Lotus semantic-depth training (YOLO-seg or COCO boxes)."""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.semantic_mask_utils import mask_path_for_image

# COCO indoor / illusion-prone objects (used when filtering detections).
DEFAULT_INDOOR_LABELS = {
    "chair", "couch", "bed", "dining table", "tv", "laptop", "book", "vase",
    "potted plant", "toilet", "sink", "refrigerator", "microwave", "oven",
    "clock", "cup", "bottle", "bowl", "remote", "keyboard", "cell phone",
    "teddy bear", "backpack", "handbag", "suitcase", "bench",
}

COCO_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck",
    9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def parse_args():
    p = argparse.ArgumentParser(description="Generate semantic masks for Hypersim RGB images.")
    p.add_argument("--rgb_dir", type=str, required=True, help="e.g. D:/lotus/data/hypersim_processed/train")
    p.add_argument("--output_dir", type=str, required=True, help="e.g. D:/lotus/data/hypersim_sem_masks")
    p.add_argument("--backend", type=str, default="yolo", choices=["yolo", "coco"])
    p.add_argument("--yolo_model", type=str, default="yolov8n-seg.pt")
    p.add_argument("--score_thr", type=float, default=0.25)
    p.add_argument("--labels", type=str, default="", help="Comma-separated COCO names; empty = all detections")
    p.add_argument("--pattern", type=str, default="rgb_cam_*.png", help="Glob for RGB files (e.g. rgb_*.png for NYUv2)")
    p.add_argument("--max_images", type=int, default=0, help="0 = all matching images under rgb_dir")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def iter_rgb_images(rgb_dir: Path, max_images: int = 0, pattern: str = "rgb_cam_*.png") -> List[Path]:
    files = sorted(rgb_dir.rglob(pattern))
    if max_images > 0:
        files = files[:max_images]
    return files


def labels_keep_set(labels_arg: str) -> Optional[Set[str]]:
    if not labels_arg.strip():
        return None
    return {x.strip().lower() for x in labels_arg.split(",") if x.strip()}


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).save(out_path)


def mask_from_yolo_result(result, h: int, w: int, labels_keep: Optional[Set[str]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    if result.masks is None or result.boxes is None:
        return mask
    names = result.names
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls.item())
        score = float(box.conf.item())
        if score < 0:
            continue
        label = str(names.get(cls_id, "")).lower()
        if labels_keep is not None and label not in labels_keep:
            continue
        seg = result.masks.data[i].detach().cpu().numpy()
        if seg.ndim == 3:
            seg = seg[0]
        seg_resized = cv2.resize(seg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.maximum(mask, seg_resized)
    return np.clip(mask, 0.0, 1.0)


def run_yolo(paths: Iterable[Path], output_dir: str, args, labels_keep: Optional[Set[str]]):
    from ultralytics import YOLO

    model = YOLO(args.yolo_model)
    for rgb_path in tqdm(list(paths), desc="YOLO-seg masks"):
        out_path = Path(mask_path_for_image(str(rgb_path), output_dir))
        if args.skip_existing and out_path.exists():
            continue
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        h, w = rgb.shape[:2]
        results = model.predict(
            source=rgb,
            imgsz=args.imgsz,
            conf=args.score_thr,
            device=args.device,
            verbose=False,
        )
        mask = mask_from_yolo_result(results[0], h, w, labels_keep)
        save_mask(mask, out_path)


def run_coco(paths: Iterable[Path], output_dir: str, args, labels_keep: Optional[Set[str]]):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()

    for rgb_path in tqdm(list(paths), desc="COCO box masks"):
        out_path = Path(mask_path_for_image(str(rgb_path), output_dir))
        if args.skip_existing and out_path.exists():
            continue
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        image_tensor = preprocess(Image.fromarray(rgb)).to(device)
        with torch.no_grad():
            outputs = model([image_tensor])[0]
        for box, score, label_idx in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score.item() < args.score_thr:
                continue
            label_name = COCO_NAMES.get(int(label_idx.item()), "").lower()
            if labels_keep is not None and label_name not in labels_keep:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = 1.0
        save_mask(mask, out_path)


def main():
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    paths = iter_rgb_images(rgb_dir, max_images=args.max_images, pattern=args.pattern)
    if not paths:
        raise FileNotFoundError(f"No {args.pattern} found under {rgb_dir}")

    labels_keep = labels_keep_set(args.labels)
    if labels_keep is None and args.backend == "coco":
        labels_keep = DEFAULT_INDOOR_LABELS

    print(f"[generate_semantic_masks] images={len(paths)} backend={args.backend}")
    print(f"[generate_semantic_masks] output={args.output_dir}")

    if args.backend == "yolo":
        run_yolo(paths, args.output_dir, args, labels_keep)
    else:
        run_coco(paths, args.output_dir, args, labels_keep)

    print("[generate_semantic_masks] done")


if __name__ == "__main__":
    main()
