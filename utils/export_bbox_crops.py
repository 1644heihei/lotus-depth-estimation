"""Export YOLO bbox crops from NYUv2 test set (no depth inference).

Usage:
  python utils/export_bbox_crops.py --output_dir output/object_predepth_bbox
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.dataset_depth import DatasetMode, get_dataset
from utils.object_detection import YoloBboxDetector, parse_label_filter
from utils.semantic_mask_utils import expand_box


def parse_args():
    p = argparse.ArgumentParser(description="Export bbox crops from NYUv2 test images.")
    p.add_argument("--dataset_config", type=str, default="datasets/eval/depth/configs/data_nyu_test.yaml")
    p.add_argument("--base_data_dir", type=str, default="datasets/eval/depth")
    p.add_argument("--output_dir", type=str, default="output/object_predepth_bbox")
    p.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    p.add_argument("--yolo_score_thr", type=float, default=0.25)
    p.add_argument("--yolo_labels", type=str, default="")
    p.add_argument("--yolo_device", type=str, default="0")
    p.add_argument("--roi_min_area", type=int, default=500)
    p.add_argument("--roi_expand_ratio", type=float, default=0.25)
    p.add_argument("--max_images", type=int, default=0, help="0 = all test images.")
    return p.parse_args()


def main():
    args = parse_args()
    out_crops = os.path.join(args.output_dir, "object_crops")
    os.makedirs(out_crops, exist_ok=True)

    bbox_detector = YoloBboxDetector(
        model_path=args.yolo_model,
        score_thr=args.yolo_score_thr,
        labels_keep=parse_label_filter(args.yolo_labels),
        device=args.yolo_device,
    )

    cfg_data = OmegaConf.load(args.dataset_config)
    dataset = get_dataset(cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.EVAL)

    num_samples = len(dataset)
    if args.max_images > 0:
        num_samples = min(num_samples, args.max_images)

    total_crops = 0
    images_with_crops = 0

    for idx in tqdm(range(num_samples), desc="Export bbox crops"):
        data = dataset[idx]
        rgb_np = data["rgb_int"].numpy().transpose(1, 2, 0).astype(np.uint8)
        rel_path = data["rgb_relative_path"]
        h, w = rgb_np.shape[:2]
        stem = rel_path.replace("/", "_").replace("\\", "_").removesuffix(".png")

        raw_boxes = bbox_detector.detect(rgb_np, min_area=args.roi_min_area)
        saved = 0
        for i, box in enumerate(raw_boxes):
            x1, y1, x2, y2 = expand_box(box, (h, w), expand_ratio=args.roi_expand_ratio)
            crop = rgb_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            Image.fromarray(crop).save(os.path.join(out_crops, f"{stem}_bbox{i:02d}.jpg"))
            saved += 1

        if saved > 0:
            images_with_crops += 1
            total_crops += saved

    print(
        f"Saved {total_crops} crops from {images_with_crops}/{num_samples} images -> {out_crops}"
    )


if __name__ == "__main__":
    main()
