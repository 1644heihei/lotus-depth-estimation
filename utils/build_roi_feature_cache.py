#!/usr/bin/env python
"""Build per-image ROI visual feature cache for object depth regression."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.detail_train_dataset import list_rgb_images, load_detail_train_manifest
from utils.object_depth_record import load_object_records_for_rgb, objects_csv_path
from utils.roi_feature_extractor import RoiFeatureExtractor
from utils.seed_all import seed_all

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build ROI feature cache (.npz) for object depth regression.")
    parser.add_argument("--records_root", type=str, required=True, help="Root dir of *_objects.csv files.")
    parser.add_argument("--output", type=str, required=True, help="Output .npz cache file path.")
    parser.add_argument("--manifest", type=str, default=None, help="JSON manifest containing rgb_paths.")
    parser.add_argument("--rgb_dir", type=str, default=None, help="Root dir of RGB images (if no manifest).")
    parser.add_argument("--pattern", type=str, default="rgb_cam_*.png", help="Pattern to match RGB images.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for ROI feature extraction.")
    parser.add_argument("--fp16", action="store_true", default=True, help="Save features in float16 to save space.")
    parser.add_argument("--max_images", type=int, default=0, help="Max images to process (0 = all).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_image_paths(args) -> List[str]:
    if args.manifest:
        rgb_paths, _, _ = load_detail_train_manifest(args.manifest)
        paths = list(rgb_paths)
    elif args.rgb_dir:
        paths = [str(p) for p in list_rgb_images(args.rgb_dir, args.pattern)]
    else:
        raise ValueError("Must provide either --manifest or --rgb_dir")
    if args.max_images > 0:
        paths = paths[: args.max_images]
    return paths


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    paths = collect_image_paths(args)
    if not paths:
        raise FileNotFoundError("No RGB image paths found.")

    logger.info("Initializing RoiFeatureExtractor on device=%s ...", args.device)
    device = torch.device(args.device)
    extractor = RoiFeatureExtractor(device=device, dtype=torch.float32)

    image_keys: List[str] = []
    offsets: List[int] = [0]
    all_features_list: List[np.ndarray] = []

    total_objects = 0
    skipped_images = 0

    for rgb_path in tqdm(paths, desc="Extracting ROI features", mininterval=10.0):
        records = load_object_records_for_rgb(rgb_path, args.records_root)
        if not records:
            # Check if csv exists
            csv_p = objects_csv_path(rgb_path, args.records_root)
            if not csv_p.is_file():
                skipped_images += 1
                continue

        image_keys.append(str(rgb_path).replace("\\", "/"))
        if not records:
            offsets.append(offsets[-1])
            continue

        try:
            with Image.open(rgb_path) as img:
                img_rgb = np.array(img.convert("RGB"))
        except Exception as e:
            logger.warning("Could not read image %s: %s", rgb_path, e)
            offsets.append(offsets[-1])
            continue

        bboxes = [[r.x1, r.y1, r.x2, r.y2] for r in records]
        feats = extractor.extract_batch(img_rgb, bboxes, batch_size=args.batch_size)
        if args.fp16:
            feats = feats.astype(np.float16)

        all_features_list.append(feats)
        total_objects += len(records)
        offsets.append(offsets[-1] + len(records))

    if all_features_list:
        combined_features = np.concatenate(all_features_list, axis=0)
    else:
        combined_features = np.empty((0, extractor.feature_dim), dtype=np.float16 if args.fp16 else np.float32)

    offsets_arr = np.asarray(offsets, dtype=np.int64)
    image_keys_arr = np.asarray(image_keys)

    metadata = {
        "schema_version": 1,
        "backbone": "resnet18",
        "feature_dim": extractor.feature_dim,
        "crop_size": extractor.target_size,
        "fp16": bool(args.fp16),
        "total_images": len(image_keys),
        "total_objects": total_objects,
        "skipped_images": skipped_images,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        image_keys=image_keys_arr,
        offsets=offsets_arr,
        roi_features=combined_features,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    logger.info("Saved ROI cache to %s", out_path)
    logger.info("Metadata: %s", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
