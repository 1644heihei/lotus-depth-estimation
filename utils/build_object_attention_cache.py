#!/usr/bin/env python
"""Build a compact per-image cache for regressor object attention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.detail_train_dataset import list_rgb_images, load_detail_train_manifest
from utils.object_attention_condition import (
    object_rows_from_detections,
    stack_object_rows,
)
from utils.object_detection_cache import load_detections
from utils.object_depth_regressor import ObjectDepthRegressorBundle
from utils.seed_all import seed_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--detections_root", required=True)
    parser.add_argument("--regressor_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pattern", default="rgb_cam_*.png")
    parser.add_argument("--detection_score_thr", type=float, default=None)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_paths(args) -> list[str]:
    if args.manifest:
        rgb_paths, _, _ = load_detail_train_manifest(args.manifest)
        paths = list(rgb_paths)
    else:
        paths = [str(path) for path in list_rgb_images(args.rgb_dir, args.pattern)]
    return paths[: args.max_images] if args.max_images > 0 else paths


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regressor = ObjectDepthRegressorBundle.load(args.regressor_dir, device=device)
    threshold = (
        float(args.detection_score_thr)
        if args.detection_score_thr is not None
        else float(regressor.config.get("detection_score_thr", 0.5))
    )
    paths = image_paths(args)
    if not paths:
        raise FileNotFoundError("No RGB images found")

    rows = []
    object_count = 0
    use_roi = regressor.config.get("roi_feature_dim", 0) > 0
    for rgb_path in tqdm(paths, desc="object_attention_cache", mininterval=30.0):
        with Image.open(rgb_path) as image:
            width, height = image.size
            rgb_np = np.array(image.convert("RGB")) if use_roi else None
        detections = [
            detection
            for detection in load_detections(rgb_path, args.detections_root)
            if detection.score >= threshold
        ]
        detections.sort(key=lambda detection: detection.score, reverse=True)
        class_ids, features = object_rows_from_detections(
            detections, width, height, regressor, rgb_np=rgb_np
        )
        object_count += len(class_ids)
        rows.append((rgb_path, class_ids, features))

    image_keys, offsets, class_ids, features = stack_object_rows(rows)
    metadata = {
        "schema_version": 1,
        "feature_dim": int(features.shape[1]),
        "regressor_dir": str(Path(args.regressor_dir)),
        "regressor_feature_version": regressor.config.get("feature_version"),
        "regressor_model_type": regressor.config.get("model_type"),
        "roi_feature_dim": int(regressor.config.get("roi_feature_dim", 0) or 0),
        "detection_score_thr": threshold,
        "num_images": len(paths),
        "num_objects": object_count,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image_keys=image_keys,
        offsets=offsets,
        class_ids=class_ids,
        features=features,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
