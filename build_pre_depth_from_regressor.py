#!/usr/bin/env python
"""Offline batch: build regressor-based pre-depth artifacts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.detail_train_dataset import list_rgb_images
from utils.object_depth_regressor import ObjectDepthRegressorBundle
from utils.object_detection_cache import load_detections
from utils.object_pre_depth import CoreDepthPredictor, save_pre_depth_artifacts
from utils.object_pre_depth_regressor import build_pre_depth_from_regressor
from utils.seed_all import seed_all
from pipeline import LotusDPipeline
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Build pre-depth from object depth regressor.")
    p.add_argument("--rgb_dir", type=str, required=True)
    p.add_argument("--detail_root", type=str, required=True, help="YOLO detections root")
    p.add_argument("--output_dir", type=str, required=True, help="Where to write pre_depth/valid_mask")
    p.add_argument("--regressor_dir", type=str, required=True)
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--pattern", type=str, default="rgb_*.png")
    p.add_argument("--detection_score_thr", type=float, default=None)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--storage_res", type=int, default=512)
    p.add_argument("--full_precision_artifacts", action="store_true")
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    rgb_paths = [str(p) for p in list_rgb_images(args.rgb_dir, args.pattern)]
    if args.max_images > 0:
        rgb_paths = rgb_paths[: args.max_images]

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regressor = ObjectDepthRegressorBundle.load(args.regressor_dir, device=device)
    detection_score_thr = (
        float(args.detection_score_thr)
        if args.detection_score_thr is not None
        else float(regressor.config.get("detection_score_thr", 0.5))
    )
    processing_res = (
        args.processing_res
        if args.processing_res is not None
        else regressor.config.get("processing_res")
    )
    logging.info(
        "Runtime config detection_score_thr=%.3f processing_res=%s precision=%s",
        detection_score_thr,
        processing_res,
        dtype,
    )
    pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    predictor = CoreDepthPredictor(
        pipe,
        processing_res=processing_res,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    from utils.object_pre_depth import pre_depth_path

    stats = {"written": 0, "skipped": 0}
    pending_paths = []
    for rgb_path in rgb_paths:
        if args.skip_existing and pre_depth_path(rgb_path, args.output_dir).is_file():
            stats["skipped"] += 1
            continue
        pending_paths.append(rgb_path)

    progress = tqdm(
        total=len(pending_paths), desc="regressor_predepth", mininterval=30.0
    )
    for start in range(0, len(pending_paths), args.batch_size):
        batch_paths = pending_paths[start : start + args.batch_size]
        rgb_batch = [
            np.array(Image.open(rgb_path).convert("RGB")) for rgb_path in batch_paths
        ]
        if len({rgb.shape for rgb in rgb_batch}) == 1:
            global_batch = predictor.predict_rgb_batch(rgb_batch)
        else:
            global_batch = [predictor.predict_rgb(rgb) for rgb in rgb_batch]
        for rgb_path, rgb_np, global_depth in zip(
            batch_paths, rgb_batch, global_batch
        ):
            detections = [
                d
                for d in load_detections(rgb_path, args.detail_root)
                if d.score >= detection_score_thr
            ]
            pre_depth, valid_mask = build_pre_depth_from_regressor(
                global_depth, detections, regressor
            )
            save_pre_depth_artifacts(
                pre_depth,
                valid_mask,
                rgb_path,
                args.output_dir,
                storage_size=(args.storage_res, args.storage_res),
                compact=not args.full_precision_artifacts,
            )
            stats["written"] += 1
            progress.update(1)
    progress.close()

    logging.info("Done stats=%s output=%s", stats, args.output_dir)


if __name__ == "__main__":
    main()
