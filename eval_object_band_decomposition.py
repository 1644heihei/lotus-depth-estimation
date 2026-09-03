#!/usr/bin/env python
"""Where inside the object region does the recoverable error actually sit?

The controlled object oracle (docs/object_oracle_control.md) left two facts to reconcile:
correcting an object's depth LEVEL is worth nothing net of control (A_bbox -0.60%), while
replacing the region outright is worth +3.91% (bbox) / +2.55% (mask). So the residue is
not the level. But the bbox beats the silhouette despite the control matching area, which
is backwards for a "the object's shape carries the signal" story - the bbox's extra 6.4%
of image area is background around the object, not object.

So split the region into bands and score each one alone, each against its own relocated
control:

  interior   erode(mask, k)                     - the object's own body
  boundary   dilate(mask,k) minus erode(mask,k) - the silhouette +-k px, where depth
                                                  discontinuities live
  collar     bbox minus dilate(mask,k)          - inside the rectangle, off the object

Whichever band carries the net gain decides whether object detection is the right
instrument here at all: boundary keeps the direction alive (detection supplies
discontinuity locations), collar kills it (the error is background, unrelated to objects).

Reuses the prediction and YOLO-seg caches, so this is CPU-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_object_oracle_ceiling import (
    _cache_path,
    align_to_gt,
    bbox_instance_masks,
    load_or_build_masks,
    oracle_replace,
    score,
)
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from utils.object_detection_cache import load_detections

BANDS = ["interior", "boundary", "collar", "bbox_all", "mask_all"]


def parse_args():
    p = argparse.ArgumentParser(description="Band decomposition of the object-region oracle.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--detail_artifacts_dir", type=str, default="D:/lotus/data/nyuv2_detail_artifacts/test")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/yolo_seg")
    p.add_argument("--output_dir", type=str, default="output/eval_object_bands")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--band_px", type=int, default=8, help="Half-width k of the boundary band.")
    p.add_argument("--n_controls", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_artifacts_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    mask_cache = Path(args.mask_cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]
    logging.info("Images: %d   band half-width: %d px", len(pairs), args.band_px)

    k = 2 * args.band_px + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    acc = {b: {"real": 0.0, "ctrl": 0.0, "area": 0.0} for b in BANDS}
    base_sum, n = 0.0, 0
    rows = []

    for rgb_path, depth_path in tqdm(pairs, desc="bands"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        if valid.sum() < 100:
            continue
        pred_path = _cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        if not pred_path.is_file():
            continue
        depth_base = align_to_gt(np.load(pred_path).astype(np.float64), gt, valid)
        if depth_base is None:
            continue

        dets = [d for d in load_detections(rgb_path, detail_root) if d.score >= args.detection_score_thr]
        bbox_u = bbox_instance_masks(dets, h, w)
        bbox_u = np.any(np.stack(bbox_u), axis=0) if bbox_u else np.zeros((h, w), bool)
        # yolo=None: masks must already be cached. The RGB is only read for its shape, so
        # a stand-in of the right size avoids decoding 654 images we do not otherwise need.
        seg = list(load_or_build_masks(rgb_path, rgb_dir, mask_cache, None,
                                       np.empty((h, w, 3), np.uint8), args.detection_score_thr))
        seg_u = np.any(np.stack(seg), axis=0) if seg else np.zeros((h, w), bool)
        if not seg_u.any() and not bbox_u.any():
            continue

        u8 = seg_u.astype(np.uint8)
        inner = cv2.erode(u8, ker).astype(bool)
        outer = cv2.dilate(u8, ker).astype(bool)
        bands = {
            "interior": inner,
            "boundary": outer & ~inner,
            "collar": bbox_u & ~outer,
            "bbox_all": bbox_u,
            "mask_all": seg_u,
        }

        b0 = score(depth_base, gt, valid)[0]
        base_sum += b0
        n += 1
        row = {"filename": str(rgb_path.relative_to(rgb_dir)).replace("\\", "/")}

        for name, reg in bands.items():
            area = float((reg & valid).sum() / max(valid.sum(), 1))
            real = score(oracle_replace(depth_base, gt, reg, valid), gt, valid)[0]
            ctrl = 0.0
            for _ in range(args.n_controls):
                dy = int(rng.integers(h // 5, 4 * h // 5))
                dx = int(rng.integers(w // 5, 4 * w // 5))
                ctrl += score(
                    oracle_replace(depth_base, gt, np.roll(reg, (dy, dx), axis=(0, 1)), valid),
                    gt, valid,
                )[0]
            ctrl /= args.n_controls
            acc[name]["real"] += real
            acc[name]["ctrl"] += ctrl
            acc[name]["area"] += area
            row[f"area_{name}"] = area
            row[f"real_{name}"] = real
            row[f"ctrl_{name}"] = ctrl
        rows.append(row)

    base = base_sum / n
    summary = {"n_images": n, "band_px": args.band_px, "baseline_abs_rel": base, "bands": {}}
    for b, e in acc.items():
        gr = 100.0 * (base - e["real"] / n) / base
        gc = 100.0 * (base - e["ctrl"] / n) / base
        area = e["area"] / n
        summary["bands"][b] = {
            "area_frac": area, "gain_real": gr, "gain_ctrl": gc, "gain_net": gr - gc,
            "net_per_area": (gr - gc) / area if area > 1e-9 else 0.0,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with open(out_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)

    print(f"\nBand decomposition  n={n}  baseline abs_rel={base:.5f}  band=+-{args.band_px}px")
    print(f"\n{'band':<10}{'area':>8}{'real':>9}{'control':>9}{'NET':>9}{'net/area':>10}")
    print("-" * 55)
    for b in BANDS:
        v = summary["bands"][b]
        print(f"{b:<10}{v['area_frac']*100:>7.1f}%{v['gain_real']:>8.2f}%"
              f"{v['gain_ctrl']:>8.2f}%{v['gain_net']:>8.2f}%{v['net_per_area']:>9.2f}")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
