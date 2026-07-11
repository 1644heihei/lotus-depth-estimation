import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.semantic_mask_utils import load_mask, resolve_mask_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compute global/ROI/boundary depth metrics.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing *.npy predictions.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing *.npy GT depth maps.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing object masks.")
    parser.add_argument("--baseline_pred_dir", type=str, default=None, help="Optional baseline prediction dir.")
    parser.add_argument("--class_map_json", type=str, default=None, help='Optional JSON { "img_stem": "class_name" }.')
    parser.add_argument("--boundary_width", type=int, default=8)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def _load_npy_depth(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return np.clip(arr, 1e-6, None)


def _compute_metrics(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    pred_t = torch.from_numpy(pred).float()
    gt_t = torch.from_numpy(gt).float()
    vm_t = torch.from_numpy(valid_mask.astype(np.bool_))
    n = vm_t.sum().clamp_min(1).float()

    absrel = (torch.abs(pred_t - gt_t) / gt_t.clamp_min(1e-6))[vm_t].sum() / n
    rmse = torch.sqrt((((pred_t - gt_t) ** 2)[vm_t].sum()) / n)
    ratio = torch.maximum(pred_t / gt_t.clamp_min(1e-6), gt_t / pred_t.clamp_min(1e-6))
    delta1 = (ratio[vm_t] < 1.25).float().mean()
    return {
        "abs_relative_difference": float(absrel.item()),
        "rmse_linear": float(rmse.item()),
        "delta1_acc": float(delta1.item()),
    }


def _boundary_mask(mask: np.ndarray, width: int) -> np.ndarray:
    k = np.ones((max(1, width), max(1, width)), dtype=np.uint8)
    dil = cv2.dilate((mask > 0.5).astype(np.uint8), k, iterations=1)
    ero = cv2.erode((mask > 0.5).astype(np.uint8), k, iterations=1)
    boundary = (dil - ero) > 0
    return boundary


def _mean_dict(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    class_map: Dict[str, str] = {}
    if args.class_map_json:
        with open(args.class_map_json, "r", encoding="utf-8") as f:
            class_map = json.load(f)

    pred_paths = sorted(Path(args.pred_dir).glob("*.npy"))
    rows = []
    class_rows = defaultdict(list)
    roi_improvements = []

    for pred_path in pred_paths:
        stem = pred_path.stem
        gt_path = os.path.join(args.gt_dir, f"{stem}.npy")
        if not os.path.exists(gt_path):
            continue
        mask_path = resolve_mask_path(stem, args.mask_dir) if os.path.isdir(args.mask_dir) else None
        if mask_path is None:
            # resolve by f"{stem}.ext" convention
            mask_path = resolve_mask_path(f"{stem}.png", args.mask_dir)
        if mask_path is None:
            continue

        pred = _load_npy_depth(str(pred_path))
        gt = _load_npy_depth(gt_path)
        mask = load_mask(mask_path)
        if mask.shape != gt.shape:
            mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        valid_global = gt > 1e-6
        valid_roi = valid_global & (mask > 0.5)
        valid_boundary = valid_global & _boundary_mask(mask, width=args.boundary_width)
        if valid_roi.sum() == 0:
            continue

        global_m = _compute_metrics(pred, gt, valid_global)
        roi_m = _compute_metrics(pred, gt, valid_roi)
        boundary_m = _compute_metrics(pred, gt, valid_boundary if valid_boundary.sum() > 0 else valid_roi)

        row = {
            "image": stem,
            "class_name": class_map.get(stem, "unknown"),
            "global_absrel": global_m["abs_relative_difference"],
            "roi_absrel": roi_m["abs_relative_difference"],
            "boundary_absrel": boundary_m["abs_relative_difference"],
            "global_rmse": global_m["rmse_linear"],
            "roi_rmse": roi_m["rmse_linear"],
            "boundary_rmse": boundary_m["rmse_linear"],
            "global_delta1": global_m["delta1_acc"],
            "roi_delta1": roi_m["delta1_acc"],
            "boundary_delta1": boundary_m["delta1_acc"],
        }

        if args.baseline_pred_dir:
            baseline_path = os.path.join(args.baseline_pred_dir, f"{stem}.npy")
            if os.path.exists(baseline_path):
                baseline_pred = _load_npy_depth(baseline_path)
                base_roi = _compute_metrics(baseline_pred, gt, valid_roi)
                row["baseline_roi_absrel"] = base_roi["abs_relative_difference"]
                row["roi_absrel_improvement"] = base_roi["abs_relative_difference"] - roi_m["abs_relative_difference"]
                roi_improvements.append(row["roi_absrel_improvement"])
        rows.append(row)
        class_rows[row["class_name"]].append(row)

    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else []
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_samples": len(rows),
        "overall": _mean_dict([{k: v for k, v in r.items() if isinstance(v, (int, float))} for r in rows]) if rows else {},
        "per_class": {
            class_name: _mean_dict([{k: v for k, v in r.items() if isinstance(v, (int, float))} for r in crow])
            for class_name, crow in class_rows.items()
        },
    }
    if roi_improvements:
        summary["roi_absrel_improvement_mean"] = float(np.mean(roi_improvements))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved per-sample metrics: {args.output_csv}")
    print(f"Saved summary: {args.output_json}")


if __name__ == "__main__":
    main()
