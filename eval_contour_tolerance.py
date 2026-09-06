#!/usr/bin/env python
"""How accurate does an estimated contour have to be before the gain disappears?

A perfect contour, used only as a barrier and with Lotus's own depth values, is worth
+80.3% BF1 net of control (eval_perfect_contour_ceiling.py). That is an oracle. Whether it
describes a method depends on how fast the gain decays as the contour moves, and there is a
number to compare against: YOLO-seg contours sit a median 2.00px from the true
discontinuity on Hypersim, once the 42.9% that mark no depth step are excluded.

  gain survives 2px  -> reachable with detectors already in hand
  gain dies by 2px   -> only a perfect contour works, and that is not a method

Displacement is a smooth random field rather than a global shift or per-pixel jitter. An
estimator's error is locally coherent - a contour drifts along a stretch, it does not
scatter - and a global shift would understate the damage by keeping the contour's shape
intact everywhere at once. The realised displacement is measured rather than assumed, so
the axis means what it says.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_boundary_f1 import boundary_f1
from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path, align_to_gt, score
from eval_perfect_contour_ceiling import propagate_labels, refill_from_own_side
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs


def parse_args():
    p = argparse.ArgumentParser(description="Gain versus contour accuracy.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--output_dir", type=str, default="output/eval_contour_tolerance")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--band_px", type=int, default=2)
    p.add_argument("--fill_radius", type=int, default=7)
    p.add_argument("--displacements", type=float, nargs="+", default=[0, 1, 2, 3, 4, 6, 8])
    p.add_argument("--field_sigma", type=float, default=24.0,
                   help="Smoothness of the displacement field, in pixels.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def smooth_displacement(shape, magnitude, sigma, rng):
    """A locally coherent random warp whose mean displacement is `magnitude` pixels."""
    h, w = shape
    dx = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    norm = np.sqrt(dx * dx + dy * dy).mean()
    if norm < 1e-9:
        return np.zeros_like(dx), np.zeros_like(dy)
    s = magnitude / norm
    return dx * s, dy * s


def warp_mask(mask, dx, dy):
    h, w = mask.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap((mask.astype(np.uint8) * 255), gx + dx, gy + dy,
                     interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    thresholds = np.linspace(5.0, 25.0, 11)
    weights = thresholds / thresholds.sum()
    ker = np.ones((2 * args.band_px + 1, 2 * args.band_px + 1), np.uint8)
    levels = list(args.displacements)
    bf1 = {k: [] for k in levels}
    ar = {k: [] for k in levels}
    realised = {k: [] for k in levels}
    base_bf1, base_ar = [], []

    def weighted(curve):
        ok = np.isfinite(curve)
        return float((curve[ok] * weights[ok]).sum() / weights[ok].sum()) if ok.any() else np.nan

    for rgb_path, depth_path in tqdm(pairs, desc="tolerance"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        if valid.sum() < 100:
            continue
        pp = _cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        if not pp.is_file():
            continue
        base = align_to_gt(np.load(pp).astype(np.float64), gt, valid)
        if base is None:
            continue
        disc = discontinuities(gt, valid, args.t)
        if not disc.any():
            continue

        base_bf1.append(weighted(boundary_f1(base, gt, valid, thresholds)))
        base_ar.append(score(base, gt, valid)[0])
        d_true = ndi.distance_transform_edt(~disc)

        for k in levels:
            if k <= 0:
                used = disc
            else:
                dx, dy = smooth_displacement((h, w), k, args.field_sigma, rng)
                used = warp_mask(disc, dx, dy) & valid
                if not used.any():
                    used = disc
            realised[k].append(float(np.median(d_true[used])))

            band = cv2.dilate(used.astype(np.uint8), ker).astype(bool) & valid
            labels, _ = ndi.label(valid & ~band)
            labels = propagate_labels(labels, band, max_iter=args.band_px + 2)
            out = refill_from_own_side(base, labels, band, args.fill_radius)
            bf1[k].append(weighted(boundary_f1(out, gt, valid, thresholds)))
            ar[k].append(score(out, gt, valid)[0])

    b_bf1, b_ar = float(np.nanmean(base_bf1)), float(np.mean(base_ar))
    summary = {"n_images": len(base_bf1), "baseline_bf1": b_bf1, "baseline_abs_rel": b_ar,
               "field_sigma": args.field_sigma, "levels": {}}
    for k in levels:
        summary["levels"][str(k)] = {
            "requested_px": k,
            "realised_median_px": float(np.mean(realised[k])),
            "bf1": float(np.nanmean(bf1[k])),
            "abs_rel": float(np.mean(ar[k])),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nContour tolerance  n={summary['n_images']}  baseline BF1={b_bf1:.4f}  "
          f"abs_rel={b_ar:.5f}")
    print(f"\n{'requested':>10}{'realised':>10}{'BF1':>9}{'vs base':>10}"
          f"{'abs_rel':>10}{'vs base':>10}")
    print("-" * 59)
    for k in levels:
        s = summary["levels"][str(k)]
        print(f"{k:>10.0f}{s['realised_median_px']:>10.2f}{s['bf1']:>9.4f}"
              f"{100*(s['bf1']-b_bf1)/b_bf1:>9.1f}%{s['abs_rel']:>10.5f}"
              f"{100*(b_ar-s['abs_rel'])/b_ar:>9.1f}%")
    print("\nYOLO-seg sits at a median 2.00px where a depth step exists (Hypersim).")
    print("The k=0 control measured -1.1%, so gains near that are worth nothing.")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
