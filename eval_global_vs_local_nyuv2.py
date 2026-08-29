#!/usr/bin/env python
"""Is Lotus's error dominantly global on NYUv2 too, or is that a Booster artifact?

On Booster, removing a 3-parameter tilt from the residual recovered 22.8% and removing
the low-frequency band recovered 75.3% - four to fourteen times what any local handle
(objects 2.1%, materials 5.3% net of control) was worth. If the same holds on NYUv2,
one property explains why every local method in this project failed, rather than five
separate dead ends.

Same decomposition as eval_booster_global_vs_local.py, run against the NYUv2 prediction
cache that eval_object_oracle_ceiling.py already built, so no inference is needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.util.alignment import align_depth_least_square, depth2disparity
from evaluation.util.metric import abs_relative_difference, delta1_acc
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from eval_booster_global_vs_local import fit_poly


def parse_args():
    p = argparse.ArgumentParser(description="Global vs local error decomposition on NYUv2.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--output_dir", type=str, default="output/eval_nyuv2_global_local")
    p.add_argument("--sigma", type=float, default=32.0)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    variants = ["baseline", "affine", "quad", "lowfreq", "planefit"]
    acc = {v: {"err": 0.0, "d1": 0.0, "n": 0} for v in variants}

    import cv2

    for rgb_path, depth_path in tqdm(pairs, desc="nyuv2_global_local"):
        rel = rgb_path.relative_to(rgb_dir)
        cpath = cache / rel.parent / f"{rel.stem}_pred.npy"
        if not cpath.is_file():
            raise FileNotFoundError(
                f"Missing cached prediction: {cpath}\n"
                f"Build it with: python eval_object_oracle_ceiling.py --processing_res "
                f"{args.processing_res} ..."
            )
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        pred = np.load(cpath).astype(np.float64)
        if pred.shape != (h, w):
            pred = np.array(
                Image.fromarray(pred.astype(np.float32)).resize((w, h), Image.BILINEAR),
                dtype=np.float64,
            )

        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        gt_disp, gt_nn = depth2disparity(depth=gt, return_mask=True)
        valid = valid & gt_nn & (pred > 0)
        if valid.sum() < 1000:
            continue

        aligned = align_depth_least_square(
            gt_arr=gt_disp, pred_arr=pred, valid_mask_arr=valid, return_scale_shift=False
        )
        resid = aligned - gt_disp

        built = {"baseline": aligned}
        built["affine"] = aligned - fit_poly(resid, valid, 1)
        built["quad"] = aligned - fit_poly(resid, valid, 2)

        r = np.where(valid, resid, 0.0).astype(np.float32)
        wgt = valid.astype(np.float32)
        ks = int(2 * round(3 * args.sigma) + 1)
        rb = cv2.GaussianBlur(r, (ks, ks), args.sigma, borderType=cv2.BORDER_REPLICATE)
        wb = cv2.GaussianBlur(wgt, (ks, ks), args.sigma, borderType=cv2.BORDER_REPLICATE)
        built["lowfreq"] = aligned - np.where(wb > 1e-3, rb / np.maximum(wb, 1e-3), 0.0)
        built["planefit"] = fit_poly(gt_disp, valid, 1)

        n = int(valid.sum())
        vt = torch.from_numpy(valid)
        gt_t = torch.from_numpy(gt)
        # Guard against an ill-conditioned fit driving corrected disparity to ~0, which
        # would invert to astronomically large depth. See the Booster script.
        floor = max(1e-3, 0.02 * float(np.median(gt_disp[valid])))
        for v, d in built.items():
            depth = np.clip(1.0 / np.clip(d, floor, None), 1e-3, 10.0)
            dt = torch.from_numpy(depth)
            acc[v]["err"] += float(abs_relative_difference(dt, gt_t, vt)) * n
            acc[v]["d1"] += float(delta1_acc(dt, gt_t, vt)) * n
            acc[v]["n"] += n

    base = acc["baseline"]["err"] / acc["baseline"]["n"]
    summary = {"processing_res": args.processing_res, "sigma": args.sigma, "variants": {}}
    for v in variants:
        a = acc[v]["err"] / acc[v]["n"]
        summary["variants"][v] = {
            "abs_rel": a,
            "delta1": acc[v]["d1"] / acc[v]["n"],
            "gain_pct": 100.0 * (base - a) / base if base else 0.0,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    label = {
        "baseline": "baseline (Lotus)",
        "affine": "remove global tilt (1st order)",
        "quad": "remove global quadratic",
        "lowfreq": f"remove low-freq (sigma={args.sigma:.0f})",
        "planefit": "single plane fitted to GT",
    }
    print(f"\nNYUv2 global-vs-local  res={args.processing_res}")
    print(f"{'variant':<34}{'abs_rel':>10}{'delta1':>9}{'gain':>9}")
    print("-" * 62)
    for v in variants:
        s = summary["variants"][v]
        print(f"{label[v]:<34}{s['abs_rel']:>10.5f}{s['delta1']:>9.4f}{s['gain_pct']:>8.1f}%")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
