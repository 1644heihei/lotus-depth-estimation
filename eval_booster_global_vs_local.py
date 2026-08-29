#!/usr/bin/env python
"""Split Lotus's Booster error into a global component and a local one.

The material oracle came back at only 2-5% net of its control, and inspecting the
worst scenes showed why: in Door - alone 19.7% of the specular/transparent error mass -
the error covers the whole frame, including the plain wall. The failure is the scene's
overall slant, not the material. The material mask simply happened to lie on top of it.

So this asks the question that observation raises: how much of the error is a smooth,
low-order surface that a single global correction could remove, and how much is genuine
local structure?

Variants, all applied post-hoc to the aligned prediction, in inverse-depth space where
the metric lives:

  affine    fit a*x + b*y + c to the residual and subtract it. Removes exactly the
            "whole scene tilted the wrong way" failure mode.
  quad      full quadratic in (x, y). A smooth global warp, still carrying no local detail.
  lowfreq   subtract a heavily blurred residual. Upper bound on any purely global fix,
            at the smoothing scale given by --sigma.
  planefit  per-scene: replace the prediction with the best-fitting single plane. Not a
            fix - a diagnostic of how nearly planar these scenes are.

Fits use only valid pixels, and the residual is measured against GT, so these are
oracles: they say what a perfect global corrector could achieve, not what a method would.
"""

from __future__ import annotations

import argparse
import csv
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

from evaluation.util.alignment import align_depth_least_square
from evaluation.util.metric import abs_relative_difference, delta1_acc
from eval_booster_mono import downsample, list_samples

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    p = argparse.ArgumentParser(description="Global vs local error decomposition on Booster.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--processing_res", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="output/eval_booster_global_local")
    p.add_argument("--illum", type=str, default="all")
    p.add_argument("--eval_scale", type=int, default=2)
    p.add_argument(
        "--sigma",
        type=float,
        default=64.0,
        help="Blur sigma (in eval-scale pixels) for the low-frequency oracle.",
    )
    p.add_argument("--max_scenes", type=int, default=0)
    return p.parse_args()


def fit_poly(residual, valid, degree):
    """Least-squares polynomial surface fitted to `residual` over `valid`."""
    h, w = residual.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx / w - 0.5).astype(np.float64)
    y = (yy / h - 0.5).astype(np.float64)
    terms = [np.ones_like(x), x, y]
    if degree >= 2:
        terms += [x * x, x * y, y * y]
    A = np.stack([t[valid] for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(A, residual[valid], rcond=None)
    return sum(c * t for c, t in zip(coef, terms))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    root = Path(args.data_root)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(root, args.illum)
    if args.max_scenes > 0:
        keep = set(sorted({s.name for s, _, _ in samples})[: args.max_scenes])
        samples = [t for t in samples if t[0].name in keep]

    variants = ["baseline", "affine", "quad", "lowfreq", "planefit"]
    acc = {v: {"err": 0.0, "d1": 0.0, "n": 0} for v in variants}
    rows = []

    import cv2

    for scene, frame, _ in tqdm(samples, desc="global_local"):
        cpath = cache / scene.name / f"{frame}_pred.npy"
        if not cpath.is_file():
            raise FileNotFoundError(f"Missing cached prediction: {cpath}")
        k = args.eval_scale
        pred = downsample(np.load(cpath).astype(np.float32), k)
        gt_disp = downsample(np.load(scene / "disp_00.npy").astype(np.float32), k)
        occ = downsample(np.array(Image.open(scene / "mask_00.png")), k, nearest=True)

        valid = np.isfinite(gt_disp) & (gt_disp > 0) & (occ > 127) & np.isfinite(pred) & (pred > 0)
        if valid.sum() < 1000:
            continue

        aligned = align_depth_least_square(
            gt_arr=gt_disp.astype(np.float64),
            pred_arr=pred.astype(np.float64),
            valid_mask_arr=valid,
            return_scale_shift=False,
        )
        gt_d = gt_disp.astype(np.float64)

        # Work in disparity (inverse depth): a planar surface is linear there, so the
        # "whole scene tilted wrong" failure is exactly a first-order term.
        resid = aligned - gt_d

        built = {"baseline": aligned}
        built["affine"] = aligned - fit_poly(resid, valid, 1)
        built["quad"] = aligned - fit_poly(resid, valid, 2)

        r = np.where(valid, resid, 0.0).astype(np.float32)
        wgt = valid.astype(np.float32)
        ks = int(2 * round(3 * args.sigma) + 1)
        rb = cv2.GaussianBlur(r, (ks, ks), args.sigma, borderType=cv2.BORDER_REPLICATE)
        wb = cv2.GaussianBlur(wgt, (ks, ks), args.sigma, borderType=cv2.BORDER_REPLICATE)
        built["lowfreq"] = aligned - np.where(wb > 1e-3, rb / np.maximum(wb, 1e-3), 0.0)

        # Diagnostic: how well does ONE plane describe the whole scene's GT?
        built["planefit"] = fit_poly(gt_d, valid, 1)

        n = int(valid.sum())
        vt = torch.from_numpy(valid)
        gt_depth = torch.from_numpy(1.0 / np.clip(gt_d, 1e-3, None))
        # A poorly conditioned polynomial fit can push corrected disparity to zero or
        # below, and inverting that produces astronomically large "depth". Clamp to a
        # data-derived floor so a bad fit degrades gracefully instead of exploding.
        floor = max(1e-3, 0.02 * float(np.median(gt_d[valid])))
        row = {"scene": scene.name, "frame": frame}
        for v, d in built.items():
            dd = torch.from_numpy(1.0 / np.clip(d, floor, None))
            a = float(abs_relative_difference(dd, gt_depth, vt))
            d1 = float(delta1_acc(dd, gt_depth, vt))
            acc[v]["err"] += a * n
            acc[v]["d1"] += d1 * n
            acc[v]["n"] += n
            row[f"absrel_{v}"] = a
        rows.append(row)

    if not rows:
        raise RuntimeError("No samples scored.")

    base = acc["baseline"]["err"] / acc["baseline"]["n"]
    summary = {
        "num_samples": len(rows),
        "processing_res": args.processing_res,
        "eval_scale": args.eval_scale,
        "sigma": args.sigma,
        "variants": {},
    }
    for v in variants:
        a = acc[v]["err"] / acc[v]["n"]
        summary["variants"][v] = {
            "abs_rel": a,
            "delta1": acc[v]["d1"] / acc[v]["n"],
            "gain_pct": 100.0 * (base - a) / base if base else 0.0,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = sorted({k for r in rows for k in r})
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    label = {
        "baseline": "baseline (Lotus)",
        "affine": "remove global tilt (1st order)",
        "quad": "remove global quadratic",
        "lowfreq": f"remove low-freq (sigma={args.sigma:.0f})",
        "planefit": "single plane fitted to GT",
    }
    print(f"\nBooster global-vs-local  res={args.processing_res}  n={len(rows)}")
    print(f"{'variant':<34}{'abs_rel':>10}{'delta1':>9}{'gain':>9}")
    print("-" * 62)
    for v in variants:
        s = summary["variants"][v]
        print(f"{label[v]:<34}{s['abs_rel']:>10.5f}{s['delta1']:>9.4f}{s['gain_pct']:>8.1f}%")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
