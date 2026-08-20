#!/usr/bin/env python
"""Evaluate object depth regressor at object level."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.detail_train_dataset import list_rgb_images
from utils.object_depth_record import load_object_records_for_rgb
from utils.object_depth_regressor import (
    ObjectDepthRegressorBundle,
    RecordWithImage,
    compute_depth_metrics,
    filter_training_items,
    image_level_split,
    load_records_from_manifest,
)
from utils.seed_all import seed_all

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate object depth regressor.")
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument(
        "--split",
        type=str,
        choices=["hypersim_holdout", "nyuv2_test", "scannet_eval"],
        required=True,
    )
    p.add_argument("--detail_root", type=str, required=True)
    p.add_argument("--manifest", type=str, default=None, help="Required for hypersim_holdout.")
    p.add_argument("--rgb_dir", type=str, default=None, help="Required for nyuv2_test.")
    p.add_argument("--pattern", type=str, default="rgb_*.png")
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_eval_items(args, bundle: ObjectDepthRegressorBundle) -> list[RecordWithImage]:
    if args.split == "hypersim_holdout":
        if not args.manifest:
            raise ValueError("--manifest required for hypersim_holdout")
        items = load_records_from_manifest(args.manifest, args.detail_root)
        depth_filter = bundle.config.get("depth_filter")
        if depth_filter:
            items = filter_training_items(
                items, min_depth_m=float(depth_filter[0]), max_depth_m=float(depth_filter[1])
            )
        split_path = Path(args.model_dir) / "split_manifest.json"
        if split_path.is_file():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            val_images = set(split["val_images"])
            val_items = [item for item in items if item.rgb_path in val_images]
        else:
            _, val_items = image_level_split(items, val_ratio=args.val_ratio, seed=args.seed)
        return val_items

    if not args.rgb_dir:
        raise ValueError("--rgb_dir required for nyuv2_test")
    rgb_paths = [str(p) for p in list_rgb_images(args.rgb_dir, args.pattern)]
    if args.max_images > 0:
        rgb_paths = rgb_paths[: args.max_images]
    out: list[RecordWithImage] = []
    for rgb in rgb_paths:
        cached = load_object_records_for_rgb(rgb, args.detail_root)
        for rec in cached:
            out.append(RecordWithImage(record=rec, rgb_path=rgb))
    return out


def metrics_by_class(items, preds, gts):
    grouped = defaultdict(list)
    for item, pred, gt in zip(items, preds, gts):
        grouped[(item.record.class_id, item.record.class_name)].append((pred, gt))
    rows = []
    for (class_id, class_name), pairs in sorted(grouped.items()):
        metrics = compute_depth_metrics(
            np.asarray([p for p, _ in pairs]), np.asarray([g for _, g in pairs])
        )
        rows.append({"class_id": class_id, "class_name": class_name, **metrics})
    return rows


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    bundle = ObjectDepthRegressorBundle.load(args.model_dir)
    baseline = bundle.baseline
    items = load_eval_items(args, bundle)
    if not items:
        raise FileNotFoundError("No evaluation records.")

    preds_mlp = []
    preds_baseline = []
    gts = []
    rows = []
    current_img_path = None
    current_img_np = None

    for it in items:
        rec = it.record
        if bundle.config.get("roi_feature_dim", 0) > 0:
            if it.rgb_path != current_img_path:
                current_img_path = it.rgb_path
                try:
                    from PIL import Image

                    with Image.open(it.rgb_path) as img:
                        current_img_np = np.array(img.convert("RGB"))
                except Exception as e:
                    logger.warning("Could not open %s: %s", it.rgb_path, e)
                    current_img_np = None
        pred = bundle.predict_record(rec, rgb_np=current_img_np)
        pred_bc = baseline.predict_depth(rec.class_id)
        preds_mlp.append(pred)
        preds_baseline.append(pred_bc)
        gts.append(rec.depth_gt_m)
        rows.append(
            {
                "rgb_path": it.rgb_path,
                "class_id": rec.class_id,
                "class_name": rec.class_name,
                "bbox_w": rec.bbox_w,
                "bbox_h": rec.bbox_h,
                "depth_gt_m": rec.depth_gt_m,
                "depth_pred_m": pred,
                "depth_pred_baseline_m": pred_bc,
                "abs_rel": abs(pred - rec.depth_gt_m) / rec.depth_gt_m,
            }
        )

    gts_arr = np.array(gts)
    selected_name = bundle.model_type
    selected_metrics = compute_depth_metrics(np.array(preds_mlp), gts_arr)
    summary = {
        "split": args.split,
        "model_dir": args.model_dir,
        "detail_root": args.detail_root,
        "num_objects": len(items),
        "selected_model_type": selected_name,
        "selected_model": selected_metrics,
        "baseline_per_class": compute_depth_metrics(np.array(preds_baseline), gts_arr),
    }
    depth_filter = bundle.config.get("depth_filter")
    if depth_filter:
        lo, hi = map(float, depth_filter)
        in_range = (gts_arr >= lo) & (gts_arr <= hi)
        summary["selected_model_in_training_range"] = compute_depth_metrics(
            np.asarray(preds_mlp)[in_range], gts_arr[in_range]
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "object_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    class_rows = metrics_by_class(items, preds_mlp, gts)
    with (out_dir / "class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Eval done: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
