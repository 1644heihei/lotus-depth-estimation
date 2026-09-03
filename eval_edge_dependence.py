#!/usr/bin/env python
"""Does Lotus only find a depth step where the image shows an edge?

docs/rgb_edge_route_closure.md measured Lotus's discontinuities as sitting closer to RGB
edges (median 1.00px) than GT's own do (1.41px), and summarised that as Lotus being
"over-attached to image edges". That summary hides a distinction the distance measurement
cannot make: a Lotus discontinuity 3px from the nearest true one might be the same step
displaced, or a different step entirely with the true one missing altogether.

Split by whether the image shows anything there and the two separate. Ground-truth
discontinuities are divided into those lying on an RGB edge and those in visually flat
regions - same-coloured furniture against a wall, an unlit doorway, a shadowless silhouette
- and recall is measured on each group. Precision splits the same way for Lotus's own
discontinuities.

  recall on edge >> recall off edge   Lotus needs a visible edge to place a depth step;
                                      invisible steps are missed rather than displaced
  recall similar on both              the failure really is displacement

Tolerance is swept (0/1/2px) because BF1's zero tolerance is what made the boundary look
catastrophic in the first place, and the shape of that curve is itself informative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path, align_to_gt
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs


def parse_args():
    p = argparse.ArgumentParser(description="Is Lotus's depth boundary dependent on image edges?")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--output_dir", type=str, default="output/eval_edge_dependence")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--canny", type=int, nargs=2, default=(50, 150))
    p.add_argument("--edge_dilate", type=int, default=1, help="Pixels of slack for 'on an edge'.")
    p.add_argument("--tolerances", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def within(mask, tol):
    """Pixels within tol of the mask (tol=0 is the mask itself)."""
    if tol <= 0:
        return mask
    k = 2 * tol + 1
    return cv2.dilate(mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def main():
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    # counts: [hit, total] per (group, tolerance)
    C = {(g, t): [0, 0] for g in ("recall_on", "recall_off", "prec_on", "prec_off")
         for t in args.tolerances}
    n_lo = n_gt = n_img = 0

    for rgb_path, depth_path in tqdm(pairs, desc="edge_dependence"):
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
        gt_d = discontinuities(gt, valid, args.t)
        lo_d = discontinuities(base, valid, args.t)
        if not gt_d.any():
            continue

        grey = np.array(Image.open(rgb_path).convert("L"))
        edge = within(cv2.Canny(grey, *args.canny).astype(bool) & valid, args.edge_dilate)

        n_img += 1
        n_gt += int(gt_d.sum())
        n_lo += int(lo_d.sum())
        groups = {
            "recall_on": (gt_d & edge, lo_d),
            "recall_off": (gt_d & ~edge, lo_d),
            "prec_on": (lo_d & edge, gt_d),
            "prec_off": (lo_d & ~edge, gt_d),
        }
        for tol in args.tolerances:
            for name, (query, target) in groups.items():
                if not query.any():
                    continue
                C[(name, tol)][0] += int((query & within(target, tol)).sum())
                C[(name, tol)][1] += int(query.sum())

    summary = {"n_images": n_img, "t": args.t, "canny": list(args.canny),
               "edge_dilate": args.edge_dilate,
               "gt_disc_px_per_image": n_gt / max(n_img, 1),
               "lotus_disc_px_per_image": n_lo / max(n_img, 1),
               "rates": {}}
    for (name, tol), (hit, tot) in C.items():
        summary["rates"][f"{name}_tol{tol}"] = {
            "rate": hit / tot if tot else 0.0, "n": tot}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    R = summary["rates"]
    print(f"\nEdge dependence  n={n_img} images  t={args.t}%  Canny{tuple(args.canny)}")
    print(f"discontinuity pixels per image: GT {summary['gt_disc_px_per_image']:.0f}   "
          f"Lotus {summary['lotus_disc_px_per_image']:.0f}   "
          f"(ratio {summary['lotus_disc_px_per_image']/max(summary['gt_disc_px_per_image'],1):.2f})")

    n_on = R[f"recall_on_tol{args.tolerances[0]}"]["n"]
    n_off = R[f"recall_off_tol{args.tolerances[0]}"]["n"]
    print(f"\nGT discontinuities: {n_on/(n_on+n_off)*100:.1f}% lie on an RGB edge, "
          f"{n_off/(n_on+n_off)*100:.1f}% in visually flat regions")

    print(f"\n--- RECALL: does Lotus find the true step? ---")
    print(f"{'tolerance':<12}{'ON an edge':>13}{'OFF an edge':>14}{'ratio':>9}")
    print("-" * 48)
    for tol in args.tolerances:
        a = R[f"recall_on_tol{tol}"]["rate"]
        b = R[f"recall_off_tol{tol}"]["rate"]
        print(f"{'+-' + str(tol) + 'px':<12}{a*100:>12.1f}%{b*100:>13.1f}%{a/max(b,1e-9):>9.2f}x")

    print(f"\n--- PRECISION: is Lotus's step a real one? ---")
    print(f"{'tolerance':<12}{'ON an edge':>13}{'OFF an edge':>14}")
    print("-" * 39)
    for tol in args.tolerances:
        print(f"{'+-' + str(tol) + 'px':<12}{R[f'prec_on_tol{tol}']['rate']*100:>12.1f}%"
              f"{R[f'prec_off_tol{tol}']['rate']*100:>13.1f}%")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
