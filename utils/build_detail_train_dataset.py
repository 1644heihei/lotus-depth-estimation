#!/usr/bin/env python
"""Build Approach-A offline detail training dataset (YOLO + core depth + class map)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import LotusDPipeline
from utils.object_condition import build_class_map_for_image, rasterize_class_map, save_class_map
from utils.object_detection_cache import (
    load_yolo_model,
    run_yolo_detections,
    save_detections,
)
from utils.object_pre_depth import CoreDepthPredictor, save_pre_depth_artifacts
from utils.seed_all import seed_all

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build offline detail training dataset (Approach A).")
    p.add_argument("--rgb_dir", type=str, required=True, help="Root directory of RGB training images")
    p.add_argument("--output_dir", type=str, required=True, help="Where to write detections/pre-depth/class maps")
    p.add_argument("--core_model", type=str, required=True, help="Core Lotus-D checkpoint or HF repo id")
    p.add_argument("--steps", type=str, default="all", help="Comma-separated: yolo,predepth,classmap,all")
    p.add_argument("--pattern", type=str, default="rgb_cam_*.png")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--yolo_model", type=str, default="yolov8n-seg.pt")
    p.add_argument("--yolo_score_thr", type=float, default=0.25)
    p.add_argument("--yolo_device", type=str, default="0")
    p.add_argument("--yolo_imgsz", type=int, default=640)
    p.add_argument("--labels", type=str, default="", help="Comma-separated COCO names; empty = all")
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--roi_min_area", type=int, default=500)
    p.add_argument("--roi_expand_ratio", type=float, default=0.25)
    p.add_argument("--align_mode", type=str, default="lstsq", choices=["lstsq", "none"])
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def iter_rgb_paths(rgb_dir: Path, pattern: str, max_images: int) -> List[Path]:
    files = sorted(rgb_dir.rglob(pattern))
    if max_images > 0:
        files = files[:max_images]
    return files


def labels_keep_set(labels_arg: str) -> Optional[Set[str]]:
    if not labels_arg.strip():
        return None
    return {x.strip().lower() for x in labels_arg.split(",") if x.strip()}


def parse_steps(steps_arg: str) -> Set[str]:
    if steps_arg.strip().lower() == "all":
        return {"yolo", "predepth", "classmap"}
    return {s.strip().lower() for s in steps_arg.split(",") if s.strip()}


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)
    steps = parse_steps(args.steps)
    labels_keep = labels_keep_set(args.labels)

    rgb_dir = Path(args.rgb_dir)
    output_dir = Path(args.output_dir)
    rgb_paths = iter_rgb_paths(rgb_dir, args.pattern, args.max_images)
    if not rgb_paths:
        raise FileNotFoundError(f"No images matching {args.pattern} under {rgb_dir}")

    yolo_model = None
    if "yolo" in steps:
        logger.info("Loading YOLO model: %s", args.yolo_model)
        yolo_model = load_yolo_model(args.yolo_model)

    predictor = None
    if "predepth" in steps:
        dtype = torch.float16 if args.half_precision else torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading core model: %s", args.core_model)
        pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        predictor = CoreDepthPredictor(
            pipe,
            timestep=args.timestep,
            processing_res=args.processing_res,
            generator=generator,
        )

    stats = {"yolo": 0, "predepth": 0, "classmap": 0, "skipped": 0}
    for rgb_path in tqdm(rgb_paths, desc="build_detail_train_dataset"):
        rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
        h, w = rgb_np.shape[:2]

        if "yolo" in steps:
            from utils.object_detection_cache import detections_json_path

            det_path = detections_json_path(rgb_path, output_dir)
            if not (args.skip_existing and det_path.is_file()):
                detections = run_yolo_detections(
                    rgb_np,
                    model=yolo_model,
                    score_thr=args.yolo_score_thr,
                    labels_keep=labels_keep,
                    imgsz=args.yolo_imgsz,
                )
                save_detections(rgb_path, output_dir, detections)
                stats["yolo"] += 1

        if "predepth" in steps:
            from utils.object_detection_cache import load_detections
            from utils.object_pre_depth import pre_depth_path

            if args.skip_existing and pre_depth_path(rgb_path, output_dir).is_file():
                stats["skipped"] += 1
            else:
                detections = load_detections(rgb_path, output_dir)
                pre_depth_norm, valid_mask, _ = predictor.build_pre_depth(
                    rgb_np,
                    detections,
                    roi_min_area=args.roi_min_area,
                    roi_expand_ratio=args.roi_expand_ratio,
                    align_mode=args.align_mode,
                )
                save_pre_depth_artifacts(pre_depth_norm, valid_mask, rgb_path, output_dir)
                stats["predepth"] += 1

        if "classmap" in steps:
            from utils.object_detection_cache import load_detections
            from utils.object_condition import class_map_path

            if args.skip_existing and class_map_path(rgb_path, output_dir).is_file():
                continue
            detections = load_detections(rgb_path, output_dir)
            class_map = rasterize_class_map(detections, h, w)
            save_class_map(class_map, rgb_path, output_dir)
            stats["classmap"] += 1

    logger.info("Done. stats=%s output_dir=%s", stats, output_dir)


if __name__ == "__main__":
    main()
