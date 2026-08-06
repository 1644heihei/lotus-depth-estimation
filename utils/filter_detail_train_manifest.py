#!/usr/bin/env python
"""Build a manifest of detail-training images that have YOLO detections."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.detail_train_dataset import (
    artifacts_ready,
    list_rgb_images,
)
from utils.object_detection_cache import load_detections

logger = logging.getLogger(__name__)


def count_detections_at_thr(rgb_path: str | Path, detail_root: str | Path, score_thr: float) -> int:
    detections = load_detections(rgb_path, detail_root)
    return sum(1 for d in detections if d.score >= score_thr)


def guess_depth_path(rgb_path: str | Path) -> str:
    p = Path(rgb_path)
    if p.name.startswith("rgb_cam_"):
        depth_name = p.name.replace("rgb_cam_", "depth_plane_cam_")
        candidate = p.parent / depth_name
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Could not infer depth path for {rgb_path}")


def collect_entries(
    rgb_paths: List[str],
    detail_root: str | Path,
    *,
    detection_score_thr: float,
    require_artifacts: bool,
) -> Tuple[List[dict], dict]:
    kept: List[dict] = []
    stats = {
        "total_scanned": 0,
        "missing_artifacts": 0,
        "zero_detections": 0,
        "with_detections": 0,
    }

    for rgb_path in rgb_paths:
        stats["total_scanned"] += 1
        if require_artifacts and not artifacts_ready(rgb_path, detail_root):
            stats["missing_artifacts"] += 1
            continue

        num_det = count_detections_at_thr(rgb_path, detail_root, detection_score_thr)
        if num_det <= 0:
            stats["zero_detections"] += 1
            continue

        depth_path = guess_depth_path(rgb_path)
        kept.append(
            {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "num_detections": num_det,
            }
        )
        stats["with_detections"] += 1

    return kept, stats


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a manifest JSON listing only images with YOLO detections."
    )
    p.add_argument("--rgb_dir", type=str, required=True, help="Root directory of RGB training images")
    p.add_argument("--detail_root", type=str, required=True, help="Offline detail artifacts root")
    p.add_argument("--output", type=str, required=True, help="Output manifest JSON path")
    p.add_argument("--pattern", type=str, default="rgb_cam_*.png")
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument(
        "--require_artifacts",
        action="store_true",
        default=True,
        help="Skip images missing pre_depth / valid_mask / class_map (default: on)",
    )
    p.add_argument(
        "--no_require_artifacts",
        action="store_false",
        dest="require_artifacts",
        help="Keep images even if offline artifacts are incomplete",
    )
    p.add_argument("--max_images", type=int, default=0, help="0 = all")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_root)
    rgb_paths = [str(p) for p in list_rgb_images(rgb_dir, args.pattern)]
    if args.max_images > 0:
        rgb_paths = rgb_paths[: args.max_images]
    if not rgb_paths:
        raise FileNotFoundError(f"No images matching {args.pattern} under {rgb_dir}")

    entries, stats = collect_entries(
        rgb_paths,
        detail_root,
        detection_score_thr=args.detection_score_thr,
        require_artifacts=args.require_artifacts,
    )

    manifest = {
        "rgb_root": str(rgb_dir.resolve()),
        "detail_root": str(detail_root.resolve()),
        "pattern": args.pattern,
        "detection_score_thr": args.detection_score_thr,
        "require_artifacts": args.require_artifacts,
        **stats,
        "entries": entries,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pct = 100.0 * stats["with_detections"] / max(stats["total_scanned"], 1)
    logger.info(
        "Manifest saved: %s  kept=%d/%d (%.1f%%)  zero_det=%d  missing_artifacts=%d",
        out_path,
        stats["with_detections"],
        stats["total_scanned"],
        pct,
        stats["zero_detections"],
        stats["missing_artifacts"],
    )


if __name__ == "__main__":
    main()
