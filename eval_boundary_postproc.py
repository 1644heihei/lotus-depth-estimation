#!/usr/bin/env python
"""How much of the boundary ceiling can training-free post-processing actually take?

docs/boundary_f1_findings.md put the object-boundary oracle at BF1 +313% net of control,
on a Lotus baseline of 0.0728 that sits at Marigold's published 0.068. That is a ceiling
with ground truth in the band; this script asks what a method with no ground truth gets.

Post-processing rather than conditioning, because every conditioning experiment so far
paid a fine-tune tax (adaptation_placement_findings.md: -4.6% to -12.6% at matched
parameter count). A ceiling worth chasing in BF1 is not worth paying that for.

Methods, all training-free:

  guided_rgb   classic RGB-guided filter, no segmentation. The control for "does the
               object mask add anything over ordinary image guidance?"
  mask_snap    the band is discarded and refilled from the nearest pixel on the SAME
               side of the YOLO-seg contour, so the blurred ramp collapses to a step
               exactly at the mask edge. Uses only Lotus's own depth plus the mask.
  snap_guided  mask_snap, then guided_rgb to soften the step's staircase artefacts.

Judged on both metrics: BF1 must rise without abs_rel degrading. A method that trades
one for the other has not fixed anything.
"""

from __future__ import annotations

import argparse
import json
import logging
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
from eval_object_oracle_ceiling import _cache_path, align_to_gt, load_or_build_masks, oracle_replace, score
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs

METHODS = ["baseline", "guided_rgb", "mask_snap", "snap_guided",
           "oracle_step", "oracle_band"]


def parse_args():
    p = argparse.ArgumentParser(description="Training-free boundary post-processing.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/yolo_seg")
    p.add_argument("--output_dir", type=str, default="output/eval_boundary_postproc")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--band_px", type=int, default=8)
    p.add_argument("--guided_radius", type=int, default=8)
    p.add_argument("--guided_eps", type=float, default=1e-4)
    p.add_argument("--t_min", type=float, default=5.0)
    p.add_argument("--t_max", type=float, default=25.0)
    p.add_argument("--n_thresholds", type=int, default=11)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def guided_filter(guide, src, radius, eps):
    """He et al. guided filter, grey guide. cv2.ximgproc is absent here, so box-mean it."""
    d = 2 * radius + 1
    ksz = (d, d)
    mean = lambda a: cv2.blur(a, ksz, borderType=cv2.BORDER_REFLECT)
    mg, ms = mean(guide), mean(src)
    cov = mean(guide * src) - mg * ms
    var = mean(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return mean(a) * guide + mean(b)


def snap_to_mask(depth, seg_u, band):
    """Collapse the blurred transition onto the mask contour.

    Band pixels are unreliable - they are where the VAE mixed foreground and background.
    Throw them away and refill each from the nearest pixel that is both outside the band
    and on the same side of the mask, so the depth steps exactly at the contour instead of
    ramping across it. No ground truth and no learning: only Lotus's own depth values,
    relocated.
    """
    out = depth.copy()
    for side in (True, False):
        target = band & (seg_u == side)
        if not target.any():
            continue
        source = (~band) & (seg_u == side)
        if not source.any():
            continue
        # nearest source pixel for every pixel, then read it only at the band
        _, (iy, ix) = ndi.distance_transform_edt(~source, return_indices=True)
        out[target] = depth[iy[target], ix[target]]
    return out


def snap_gt_levels(depth, gt, masks, band, ker, valid):
    """DIAGNOSTIC: a step of the RIGHT SIZE at the MASK's location.

    mask_snap installs a step at the mask contour using Lotus's own levels. If it fails,
    either the mask sits in the wrong place or Lotus's levels are too close together.
    This variant keeps the mask's location but takes both levels from ground truth, so
    comparing it against oracle_band (correct size AND correct location) separates the two.
    """
    out = depth.copy()
    for m in masks:
        inner = cv2.erode(m.astype(np.uint8), ker).astype(bool) & valid
        ring = (cv2.dilate(m.astype(np.uint8), ker).astype(bool) & ~m) & valid
        if inner.sum() < 50 or ring.sum() < 50:
            continue
        gi, go = float(np.median(gt[inner])), float(np.median(gt[ring]))
        if not (np.isfinite(gi) and np.isfinite(go)):
            continue
        out[band & m] = gi
        out[band & ~m & cv2.dilate(m.astype(np.uint8), ker).astype(bool)] = go
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    mask_cache = Path(args.mask_cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.linspace(args.t_min, args.t_max, args.n_thresholds)
    weights = thresholds / thresholds.sum()
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]
    logging.info("Images: %d  band=+-%dpx  guided r=%d eps=%g",
                 len(pairs), args.band_px, args.guided_radius, args.guided_eps)

    k = 2 * args.band_px + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    curves = {m: [] for m in METHODS}
    absrel = {m: [] for m in METHODS}
    contrast = []  # (predicted, gt) depth ratio across each object's contour

    for rgb_path, depth_path in tqdm(pairs, desc="postproc"):
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

        seg = list(load_or_build_masks(rgb_path, rgb_dir, mask_cache, None,
                                       np.empty((h, w, 3), np.uint8), args.detection_score_thr))
        seg_u = np.any(np.stack(seg), axis=0) if seg else np.zeros((h, w), bool)
        if not seg_u.any():
            continue
        u8 = seg_u.astype(np.uint8)
        band = cv2.dilate(u8, ker).astype(bool) & ~cv2.erode(u8, ker).astype(bool)

        rgb = np.array(Image.open(rgb_path).convert("L")).astype(np.float32) / 255.0
        b32 = base.astype(np.float32)
        g = guided_filter(rgb, b32, args.guided_radius, args.guided_eps).astype(np.float64)
        snap = snap_to_mask(base, seg_u, band)
        sg = guided_filter(rgb, snap.astype(np.float32), args.guided_radius,
                           args.guided_eps).astype(np.float64)

        out = {
            "baseline": base,
            "guided_rgb": np.clip(g, 1e-3, 10.0),
            "mask_snap": np.clip(snap, 1e-3, 10.0),
            "snap_guided": np.clip(sg, 1e-3, 10.0),
            "oracle_step": snap_gt_levels(base, gt, seg, band, ker, valid),
            "oracle_band": oracle_replace(base, gt, band, valid),
        }
        for m in seg:
            inner = cv2.erode(m.astype(np.uint8), ker).astype(bool) & valid
            ring = (cv2.dilate(m.astype(np.uint8), ker).astype(bool) & ~m) & valid
            if inner.sum() < 50 or ring.sum() < 50:
                continue
            rp = float(np.median(base[ring]) / max(np.median(base[inner]), 1e-9))
            rg = float(np.median(gt[ring]) / max(np.median(gt[inner]), 1e-9))
            contrast.append((rp, rg))

        for m, d in out.items():
            curves[m].append(boundary_f1(d, gt, valid, thresholds))
            absrel[m].append(score(d, gt, valid)[0])

    n = len(curves["baseline"])
    summary = {"n_images": n, "band_px": args.band_px,
               "guided_radius": args.guided_radius, "guided_eps": args.guided_eps,
               "thresholds": thresholds.tolist(), "methods": {}}
    for m in METHODS:
        c = np.nanmean(np.stack(curves[m]), axis=0)
        summary["methods"][m] = {"bf1": float((c * weights).sum()),
                                 "bf1_curve": c.tolist(),
                                 "abs_rel": float(np.mean(absrel[m]))}
    if contrast:
        c = np.array(contrast)
        # a contour only registers when the ratio leaves [1/(1+t/100), 1+t/100]
        dev = lambda a: np.maximum(a, 1.0 / np.maximum(a, 1e-9))
        dp, dg = dev(c[:, 0]), dev(c[:, 1])
        summary["contrast"] = {
            "n_objects": int(len(c)),
            "pred_ratio_median": float(np.median(dp)),
            "gt_ratio_median": float(np.median(dg)),
            "pred_over_gt_median": float(np.median(dp / np.maximum(dg, 1e-9))),
            "frac_detectable": {f"t{int(t)}": [float((dp > 1 + t / 100).mean()),
                                               float((dg > 1 + t / 100).mean())]
                                for t in (5, 10, 15, 25)},
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    M = summary["methods"]
    b, o = M["baseline"], M["oracle_band"]
    head = o["bf1"] - b["bf1"]
    print(f"\nTraining-free boundary post-processing  n={n}  band=+-{args.band_px}px")
    print(f"\n{'method':<14}{'BF1':>9}{'vs base':>10}{'% of ceil':>11}{'abs_rel':>10}{'vs base':>10}")
    print("-" * 64)
    for m in METHODS:
        d = M[m]
        db = 100.0 * (d["bf1"] - b["bf1"]) / b["bf1"]
        pc = 100.0 * (d["bf1"] - b["bf1"]) / head if head > 1e-9 else 0.0
        da = 100.0 * (b["abs_rel"] - d["abs_rel"]) / b["abs_rel"]
        print(f"{m:<14}{d['bf1']:>9.4f}{db:>9.1f}%{pc:>10.1f}%{d['abs_rel']:>10.5f}{da:>9.1f}%")
    print("\n('% of ceil' = share of the oracle_band headroom recovered;")
    print(" abs_rel 'vs base' positive = improved, negative = degraded)")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
