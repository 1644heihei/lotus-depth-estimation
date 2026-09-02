#!/usr/bin/env python
"""How much error would be removed by getting large planes right?

The global component holds 55-75% of the error and behaves like a tilt, which is what
getting a ground plane's orientation wrong looks like. Indoor scenes are largely floor,
walls and table tops, and those are far easier to segment than objects - which the
oracle measurements showed carry only 2-5% of recoverable error. So: if the model's
large planes were oriented correctly, how much would that buy?

Key simplification: under a pinhole camera a 3D plane projects to a LINEAR function of
image coordinates in inverse-depth (disparity) space, disp = a*u + b*v + c. Planarity
is therefore testable and correctable without knowing the intrinsics, in exactly the
basis the tilt oracle already used.

Planes are found in the GROUND TRUTH by sequential RANSAC, so this measures the ceiling
of a perfect plane-aware correction, not a method.

Variants:
  orient  within each plane's support, replace the prediction's own plane component with
          GT's and keep its local residual. The realistic target: fix the orientation,
          keep the detail.
  full    replace the support with the GT plane outright. Upper bound for the region.
  ctrl    same areas, randomly relocated. The material oracle looked strong until this
          control removed half of it, so it is included from the start.
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

from evaluation.util.alignment import align_depth_least_square
from evaluation.util.metric import abs_relative_difference, delta1_acc
from eval_booster_mono import downsample, list_samples

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    p = argparse.ArgumentParser(description="Ceiling of a plane-aware correction.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--processing_res", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="output/eval_plane_ceiling")
    p.add_argument("--illum", type=str, default="all")
    p.add_argument("--eval_scale", type=int, default=4)
    p.add_argument("--max_planes", type=int, default=4)
    p.add_argument(
        "--inlier_frac",
        type=float,
        default=0.02,
        help="Plane inlier tolerance, as a fraction of the scene's disparity range.",
    )
    p.add_argument("--min_plane_frac", type=float, default=0.03,
                   help="Discard planes covering less than this share of valid pixels.")
    p.add_argument("--ransac_iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fit_plane(u, v, d, sel):
    A = np.stack([u[sel], v[sel], np.ones(int(sel.sum()))], axis=1)
    coef, *_ = np.linalg.lstsq(A, d[sel], rcond=None)
    return coef


def eval_plane(coef, u, v):
    return coef[0] * u + coef[1] * v + coef[2]


def find_planes(d, valid, u, v, rng, tol, min_px, max_planes, iters):
    """Sequential RANSAC over (u, v, disparity). Returns [(support_mask, coef), ...]."""
    planes = []
    remaining = valid.copy()
    idx = np.flatnonzero(remaining.ravel())
    for _ in range(max_planes):
        if idx.size < max(min_px, 16):
            break
        best_in, best_coef = None, None
        for _ in range(iters):
            pick = rng.choice(idx, size=3, replace=False)
            sel = np.zeros(valid.size, bool)
            sel[pick] = True
            sel = sel.reshape(valid.shape)
            try:
                coef = fit_plane(u, v, d, sel)
            except np.linalg.LinAlgError:
                continue
            resid = np.abs(d - eval_plane(coef, u, v))
            inl = remaining & (resid < tol)
            n = int(inl.sum())
            if best_in is None or n > best_in[0]:
                best_in, best_coef = (n, inl), coef
        if best_in is None or best_in[0] < min_px:
            break
        inl = best_in[1]
        coef = fit_plane(u, v, d, inl)  # refit on all inliers
        planes.append((inl, coef))
        remaining = remaining & ~inl
        idx = np.flatnonzero(remaining.ravel())
    return planes


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.data_root)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ["baseline", "orient", "full", "orient_ctrl", "full_ctrl"]
    acc = {k: [0.0, 0.0, 0] for k in names}
    cover, nplanes, nimg = 0.0, 0.0, 0

    for scene, frame, _ in tqdm(list_samples(root, args.illum), desc="plane_ceiling"):
        cp = cache / scene.name / f"{frame}_pred.npy"
        if not cp.is_file():
            continue
        k = args.eval_scale
        pred = downsample(np.load(cp).astype(np.float32), k)
        gt = downsample(np.load(scene / "disp_00.npy").astype(np.float32), k)
        occ = downsample(np.array(Image.open(scene / "mask_00.png")), k, nearest=True)
        valid = np.isfinite(gt) & (gt > 0) & (occ > 127) & np.isfinite(pred) & (pred > 0)
        if valid.sum() < 2000:
            continue

        gtd = gt.astype(np.float64)
        aligned = align_depth_least_square(
            gt_arr=gtd, pred_arr=pred.astype(np.float64), valid_mask_arr=valid,
            return_scale_shift=False,
        )
        h, w = aligned.shape
        vv, uu = np.mgrid[0:h, 0:w]
        u = (uu / w - 0.5).astype(np.float64)
        v = (vv / h - 0.5).astype(np.float64)

        rngv = float(np.percentile(gtd[valid], 99) - np.percentile(gtd[valid], 1))
        planes = find_planes(
            gtd, valid, u, v, rng,
            tol=max(args.inlier_frac * rngv, 1e-9),
            min_px=int(args.min_plane_frac * valid.sum()),
            max_planes=args.max_planes, iters=args.ransac_iters,
        )
        if not planes:
            continue

        support = np.zeros_like(valid)
        for m, _ in planes:
            support |= m

        out_orient = aligned.copy()
        out_full = aligned.copy()
        for m, coef in planes:
            gt_plane = eval_plane(coef, u, v)
            # keep the prediction's local detail, swap only its plane component
            pc = fit_plane(u, v, aligned, m)
            pred_plane = eval_plane(pc, u, v)
            out_orient[m] = aligned[m] - pred_plane[m] + gt_plane[m]
            out_full[m] = gt_plane[m]

        # control: same-area supports, relocated. A real plane effect must beat this.
        shift = (int(rng.integers(h // 5, 4 * h // 5)), int(rng.integers(w // 5, 4 * w // 5)))
        out_o_ctrl, out_f_ctrl = aligned.copy(), aligned.copy()
        for m, _ in planes:
            ms = np.roll(m, shift, axis=(0, 1)) & valid
            if ms.sum() < 16:
                continue
            gc = fit_plane(u, v, gtd, ms)
            pc = fit_plane(u, v, aligned, ms)
            out_o_ctrl[ms] = aligned[ms] - eval_plane(pc, u, v)[ms] + eval_plane(gc, u, v)[ms]
            out_f_ctrl[ms] = eval_plane(gc, u, v)[ms]

        n = int(valid.sum())
        vt = torch.from_numpy(valid)
        gtt = torch.from_numpy(1.0 / np.clip(gtd, 1e-3, None))
        floor = max(1e-3, 0.02 * float(np.median(gtd[valid])))
        for name, arr in [("baseline", aligned), ("orient", out_orient), ("full", out_full),
                          ("orient_ctrl", out_o_ctrl), ("full_ctrl", out_f_ctrl)]:
            dd = torch.from_numpy(1.0 / np.clip(arr, floor, None))
            e = acc[name]
            e[0] += float(abs_relative_difference(dd, gtt, vt)) * n
            e[1] += float(delta1_acc(dd, gtt, vt)) * n
            e[2] += n
        cover += float((support & valid).sum() / n)
        nplanes += len(planes)
        nimg += 1

    if nimg == 0:
        raise RuntimeError("No samples scored.")

    base = acc["baseline"][0] / acc["baseline"][2]
    summary = {
        "num_images": nimg,
        "mean_plane_coverage": cover / nimg,
        "mean_planes_per_image": nplanes / nimg,
        "inlier_frac": args.inlier_frac,
        "variants": {},
    }
    for kk in names:
        a = acc[kk][0] / acc[kk][2]
        summary["variants"][kk] = {"abs_rel": a, "delta1": acc[kk][1] / acc[kk][2],
                                   "gain_pct": 100.0 * (base - a) / base}
    summary["net_orient"] = summary["variants"]["orient"]["gain_pct"] - summary["variants"]["orient_ctrl"]["gain_pct"]
    summary["net_full"] = summary["variants"]["full"]["gain_pct"] - summary["variants"]["full_ctrl"]["gain_pct"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    v = summary["variants"]
    print(f"\nPlane-correction ceiling  n={nimg}  res={args.processing_res}")
    print(f"planes/image {summary['mean_planes_per_image']:.1f}   "
          f"coverage {summary['mean_plane_coverage']*100:.1f}% of valid pixels")
    print(f"baseline abs_rel = {base:.5f}\n")
    print(f"{'variant':<16}{'abs_rel':>10}{'delta1':>9}{'gain':>9}")
    print("-" * 46)
    for kk in names:
        print(f"{kk:<16}{v[kk]['abs_rel']:>10.5f}{v[kk]['delta1']:>9.4f}{v[kk]['gain_pct']:>8.2f}%")
    print(f"\nNET of control:  orient {summary['net_orient']:+.2f}%   full {summary['net_full']:+.2f}%")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
