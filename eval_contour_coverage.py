#!/usr/bin/env python
"""What happens when the contour is incomplete rather than displaced?

eval_contour_tolerance.py perturbed the contour by moving it, and the gain collapsed within
a pixel. That is the wrong failure model for a method that TRACES an existing line - picking
contour pixels out of a mask, or following an image edge. Such a method does not slide the
line off the boundary; it misses stretches of it, and marks stretches that are not there.
Those are recall and precision failures, and they should behave completely differently:
a stretch left unmarked is simply left as Lotus had it, so partial coverage ought to
degrade gracefully instead of going negative.

That distinction decides the direction, because the measured shortfall of object contours
is coverage, not position. Where a depth step exists, YOLO-seg sits at a median 2.00px -
better than Lotus's own 3.00px. But only 7.9% of true discontinuities lie within 2px of any
object contour, since most of a room's depth edges are walls, floors, and furniture
interiors that COCO's 27 classes never name.

  RECALL      keep a fraction of the true contour, drop the rest
  PRECISION   keep all of it and add spurious contour where no step exists

Both are dropped in coherent patches rather than scattered pixels: a tracer loses a run of
boundary, it does not lose every third pixel. The scattered variant is measured too, since
the two bound the realistic case from either side.
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
    p = argparse.ArgumentParser(description="Gain versus contour coverage.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--output_dir", type=str, default="output/eval_contour_coverage")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--band_px", type=int, default=2)
    p.add_argument("--fill_radius", type=int, default=7)
    p.add_argument("--keep", type=float, nargs="+",
                   default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05])
    p.add_argument("--add", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                   help="Spurious contour added, as a multiple of the true contour's size.")
    p.add_argument("--patch_sigma", type=float, default=16.0,
                   help="Size of the coherent patches kept or dropped, in pixels.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=300)
    return p.parse_args()


def coherent_selector(shape, frac, sigma, rng):
    """A smooth random field thresholded to select `frac` of the image in patches."""
    f = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return f <= np.quantile(f, np.clip(frac, 0.0, 1.0))


def apply_contour(base, gt, valid, contour, ker, band_px, fill_radius, thresholds, weights):
    band = cv2.dilate(contour.astype(np.uint8), ker).astype(bool) & valid
    if not band.any():
        return base
    labels, _ = ndi.label(valid & ~band)
    labels = propagate_labels(labels, band, max_iter=band_px + 2)
    return refill_from_own_side(base, labels, band, fill_radius)


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

    th = np.linspace(5.0, 25.0, 11)
    wt = th / th.sum()
    ker = np.ones((2 * args.band_px + 1, 2 * args.band_px + 1), np.uint8)

    keys = ([f"keep{k}" for k in args.keep] + [f"keep{k}_scattered" for k in args.keep]
            + [f"add{a}" for a in args.add])
    bf1 = {k: [] for k in keys}
    ar = {k: [] for k in keys}
    cov = {k: [] for k in keys}
    b_bf1, b_ar = [], []

    def weighted(c):
        ok = np.isfinite(c)
        return float((c[ok] * wt[ok]).sum() / wt[ok].sum()) if ok.any() else np.nan

    for rgb_path, depth_path in tqdm(pairs, desc="coverage"):
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
        n_disc = int(disc.sum())
        if n_disc < 50:
            continue

        b_bf1.append(weighted(boundary_f1(base, gt, valid, th)))
        b_ar.append(score(base, gt, valid)[0])

        def run(key, contour):
            out = apply_contour(base, gt, valid, contour, ker, args.band_px,
                                args.fill_radius, th, wt)
            bf1[key].append(weighted(boundary_f1(out, gt, valid, th)))
            ar[key].append(score(out, gt, valid)[0])
            cov[key].append(int(contour.sum()) / max(n_disc, 1))

        for k in args.keep:
            run(f"keep{k}", disc & coherent_selector((h, w), k, args.patch_sigma, rng))
            run(f"keep{k}_scattered", disc & (rng.random((h, w)) < k))

        # Spurious contour: patches placed away from any true step, so they are wrong in
        # the way a false positive is wrong rather than being a displaced true contour.
        far = ndi.distance_transform_edt(~disc) > 6
        for a in args.add:
            n_far = int((far & valid).sum())
            if n_far < 10:
                continue
            sel = coherent_selector((h, w), min(a * n_disc / max(n_far, 1), 1.0),
                                    args.patch_sigma, rng)
            run(f"add{a}", disc | (far & valid & sel))

    n = len(b_bf1)
    B, A = float(np.nanmean(b_bf1)), float(np.mean(b_ar))
    summary = {"n_images": n, "baseline_bf1": B, "baseline_abs_rel": A, "settings": {}}
    for k in keys:
        if bf1[k]:
            summary["settings"][k] = {"bf1": float(np.nanmean(bf1[k])),
                                      "abs_rel": float(np.mean(ar[k])),
                                      "contour_frac": float(np.mean(cov[k]))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    S = summary["settings"]
    print(f"\nContour coverage  n={n}  baseline BF1={B:.4f}  abs_rel={A:.5f}")
    print(f"\nRECALL: keep a fraction of the true contour, drop the rest")
    print(f"{'kept':>8}{'coherent BF1':>15}{'vs base':>10}{'scattered BF1':>16}{'vs base':>10}")
    print("-" * 59)
    for k in args.keep:
        a, b = S.get(f"keep{k}"), S.get(f"keep{k}_scattered")
        if a and b:
            print(f"{k:>8.2f}{a['bf1']:>15.4f}{100*(a['bf1']-B)/B:>9.1f}%"
                  f"{b['bf1']:>16.4f}{100*(b['bf1']-B)/B:>9.1f}%")
    print(f"\nPRECISION: all of the true contour, plus spurious contour where no step exists")
    print(f"{'added':>8}{'BF1':>10}{'vs base':>10}{'abs_rel':>10}{'vs base':>10}")
    print("-" * 48)
    for a in args.add:
        s = S.get(f"add{a}")
        if s:
            print(f"{a:>8.2f}{s['bf1']:>10.4f}{100*(s['bf1']-B)/B:>9.1f}%"
                  f"{s['abs_rel']:>10.5f}{100*(A-s['abs_rel'])/A:>9.1f}%")
    print("\nOnly 7.9% of true discontinuities lie within 2px of an object contour,")
    print("so the recall column is where an object-based method would actually sit.")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
