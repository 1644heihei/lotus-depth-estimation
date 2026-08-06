#!/usr/bin/env python
"""Build per-object depth CSV records from YOLO detections + GT depth."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.detail_train_dataset import list_rgb_images, load_detail_train_manifest
from utils.object_depth_record import (
    build_records_for_image,
    guess_depth_path,
    objects_csv_path,
    save_object_records,
)
from utils.seed_all import seed_all

logger = logging.getLogger(__name__)
KNOWN_DATASET_DEPTH_UNITS = {
    "hypersim_train": "mm",
    "nyuv2_test": "mm",
    "scannet_eval": "mm",
}


def validate_dataset_depth_unit(dataset: str, depth_unit: str) -> None:
    expected = KNOWN_DATASET_DEPTH_UNITS.get(dataset)
    if expected is not None and depth_unit != expected:
        raise ValueError(
            f"{dataset} records expect --depth_unit={expected}, got {depth_unit}. "
            "Using the wrong unit silently corrupts every metric."
        )


def parse_args():
    p = argparse.ArgumentParser(description="Build *_objects.csv from detections + GT depth.")
    p.add_argument("--rgb_dir", type=str, required=True)
    p.add_argument("--detections_root", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--manifest", type=str, default=None, help="Optional JSON manifest of rgb paths.")
    p.add_argument("--pattern", type=str, default="rgb_cam_*.png")
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--min_pixels", type=int, default=16)
    existing = p.add_mutually_exclusive_group()
    existing.add_argument("--skip_existing", action="store_true", dest="skip_existing")
    existing.add_argument(
        "--overwrite",
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
    )
    p.set_defaults(skip_existing=True)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--dataset", type=str, required=True, help="Dataset identity stored in every record.")
    p.add_argument("--depth_unit", type=str, required=True, choices=["m", "cm", "mm"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_rgb_depth_pairs(args) -> List[tuple[str, str]]:
    if args.manifest:
        rgb_paths, depth_paths, _ = load_detail_train_manifest(args.manifest)
        pairs = list(zip(rgb_paths, depth_paths))
        if args.max_images > 0:
            pairs = pairs[: args.max_images]
        return pairs
    rgb_paths = [str(p) for p in list_rgb_images(args.rgb_dir, args.pattern)]
    if args.max_images > 0:
        rgb_paths = rgb_paths[: args.max_images]
    pairs = []
    for rgb in rgb_paths:
        pairs.append((rgb, str(guess_depth_path(rgb))))
    return pairs


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)
    validate_dataset_depth_unit(args.dataset, args.depth_unit)

    pairs = collect_rgb_depth_pairs(args)
    if not pairs:
        raise FileNotFoundError("No rgb/depth pairs to process.")

    stats = {"written": 0, "skipped": 0, "empty": 0, "total_objects": 0}
    for rgb_path, depth_path in pairs:
        out_path = objects_csv_path(rgb_path, args.output_root)
        if args.skip_existing and out_path.is_file():
            stats["skipped"] += 1
            continue
        records = build_records_for_image(
            rgb_path,
            args.detections_root,
            depth_path=depth_path,
            score_thr=args.detection_score_thr,
            min_pixels=args.min_pixels,
            dataset=args.dataset,
            depth_unit=args.depth_unit,
        )
        save_object_records(records, rgb_path, args.output_root)
        stats["written"] += 1
        stats["total_objects"] += len(records)
        if not records:
            stats["empty"] += 1

    summary = {
        "dataset": args.dataset,
        "depth_unit": args.depth_unit,
        "rgb_dir": args.rgb_dir,
        "detections_root": args.detections_root,
        "output_root": args.output_root,
        "manifest": args.manifest,
        "detection_score_thr": args.detection_score_thr,
        **stats,
        "num_images": len(pairs),
    }
    logger.info("Done. stats=%s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
