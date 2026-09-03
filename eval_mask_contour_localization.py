#!/usr/bin/env python
"""How far is a YOLO-seg silhouette from the depth discontinuity it is supposed to mark?

The post-processing run (docs/boundary_f1_findings.md, section on oracle_step) showed the
object-boundary ceiling is unreachable through segmentation masks: installing ground
truth's own depth step AT THE MASK CONTOUR halves BF1 (0.0728 -> 0.0343), while filling
the whole band with GT quadruples it (-> 0.3423). Same magnitude, different place. The
contrast diagnostic ruled out magnitude as the problem - Lotus's step across a silhouette
matches GT to 0.2%.

That leaves localisation, measured here directly, in pixels: for every mask contour pixel,
the distance to the nearest true depth discontinuity in GT, and the reverse. BF1 compares
adjacent pixel pairs and has no distance tolerance, so a median error of even a few pixels
makes the mask useless as a place to put a step.

GT discontinuities use the same ratio criterion as the metric, c = [d(j)/d(i) > 1+t/100].
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

from eval_object_oracle_ceiling import _cache_path, align_to_gt, load_or_build_masks
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs


def parse_args():
    p = argparse.ArgumentParser(description="Mask contour vs true depth discontinuity.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/yolo_seg")
    p.add_argument("--output_dir", type=str, default="output/eval_mask_localization")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--t", type=float, default=10.0, help="Discontinuity threshold, percent.")
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def discontinuities(depth, valid, t):
    """Pixels adjacent to a depth step larger than t percent."""
    r = 1.0 + t / 100.0
    out = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        b = np.roll(depth, -1, axis=axis)
        vb = valid & np.roll(valid, -1, axis=axis)
        if axis == 1:
            vb[:, -1] = False
        else:
            vb[-1, :] = False
        step = vb & ((b / np.maximum(depth, 1e-9) > r) | (depth / np.maximum(b, 1e-9) > r))
        out |= step
        out |= np.roll(step, 1, axis=axis)
    return out & valid


def main():
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    mask_to_gt, gt_to_mask, pred_to_gt = [], [], []
    for rgb_path, depth_path in tqdm(pairs, desc="localization"):
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
        seg = list(load_or_build_masks(rgb_path, rgb_dir, Path(args.mask_cache_dir), None,
                                       np.empty((h, w, 3), np.uint8), args.detection_score_thr))
        seg_u = np.any(np.stack(seg), axis=0) if seg else np.zeros((h, w), bool)
        if not seg_u.any():
            continue

        u8 = seg_u.astype(np.uint8)
        contour = (cv2.dilate(u8, np.ones((3, 3), np.uint8)).astype(bool)
                   & ~cv2.erode(u8, np.ones((3, 3), np.uint8)).astype(bool) & valid)
        gt_d = discontinuities(gt, valid, args.t)
        pr_d = discontinuities(base, valid, args.t)
        if not contour.any() or not gt_d.any():
            continue

        d_to_gt = ndi.distance_transform_edt(~gt_d)
        mask_to_gt.append(d_to_gt[contour])
        gt_to_mask.append(ndi.distance_transform_edt(~contour)[gt_d])
        if pr_d.any():
            pred_to_gt.append(d_to_gt[pr_d])

    def stats(chunks):
        a = np.concatenate(chunks)
        return {
            "n_px": int(a.size),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "frac_within_1px": float((a <= 1.0).mean()),
            "frac_within_2px": float((a <= 2.0).mean()),
            "frac_within_4px": float((a <= 4.0).mean()),
        }

    summary = {
        "t": args.t,
        "n_images": len(mask_to_gt),
        "mask_contour_to_gt_discontinuity": stats(mask_to_gt),
        "gt_discontinuity_to_mask_contour": stats(gt_to_mask),
        "lotus_discontinuity_to_gt_discontinuity": stats(pred_to_gt),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nContour localisation  n={summary['n_images']} images  t={args.t}%")
    print(f"\n{'distance from -> to':<42}{'median':>8}{'p90':>7}{'<=1px':>8}{'<=2px':>8}{'<=4px':>8}")
    print("-" * 81)
    for label, key in (
        ("YOLO mask contour -> GT discontinuity", "mask_contour_to_gt_discontinuity"),
        ("GT discontinuity -> YOLO mask contour", "gt_discontinuity_to_mask_contour"),
        ("Lotus discontinuity -> GT discontinuity", "lotus_discontinuity_to_gt_discontinuity"),
    ):
        s = summary[key]
        print(f"{label:<42}{s['median']:>8.2f}{s['p90']:>7.1f}"
              f"{s['frac_within_1px']*100:>7.1f}%{s['frac_within_2px']*100:>7.1f}%"
              f"{s['frac_within_4px']*100:>7.1f}%")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
