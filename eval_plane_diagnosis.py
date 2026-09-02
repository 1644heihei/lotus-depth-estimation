#!/usr/bin/env python
"""Does Lotus know WHERE the planes are, and get their ORIENTATION wrong?

Correcting large planes to GT is worth +23-29% net of control - more than any other
sensor-free handle measured. But that oracle hands over the GT plane parameters, and a
method has to estimate them. Which part is actually missing decides what to build, and
whether it differs from P3Depth (CVPR 2022), which already predicts per-pixel plane
coefficients:

  supports agree, orientations wrong -> the model already segments planes; only the
      geometry needs fixing, and a light correction suffices
  supports disagree -> plane detection is needed too, and the work collapses into
      what P3Depth already does

Also splits the correction's value by parameter. A plane is disp = a*u + b*v + c, so
(a, b) is its orientation and c its offset. If most of the gain comes from c, the
problem is a per-region offset - far easier than inferring orientation.

Planes come from RANSAC on GT and, separately, on the aligned prediction, so support
agreement is measured between two independent detections.
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
from evaluation.util.metric import abs_relative_difference
from eval_booster_mono import downsample, list_samples
from eval_plane_ceiling import eval_plane, find_planes, fit_plane

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose plane support vs orientation error.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--processing_res", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="output/eval_plane_diagnosis")
    p.add_argument("--illum", type=str, default="im0")
    p.add_argument("--eval_scale", type=int, default=4)
    p.add_argument("--max_planes", type=int, default=4)
    p.add_argument("--inlier_frac", type=float, default=0.008)
    p.add_argument("--min_plane_frac", type=float, default=0.03)
    p.add_argument("--ransac_iters", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.data_root)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["baseline", "fix_c", "fix_ab", "fix_all"]
    acc = {k: [0.0, 0] for k in variants}
    ious, ang_err, rel_ab, rel_c, tilt_mag = [], [], [], [], []
    nimg = 0

    for scene, frame, _ in tqdm(list_samples(root, args.illum), desc="plane_diagnosis"):
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
        tol = max(args.inlier_frac * rngv, 1e-9)
        minpx = int(args.min_plane_frac * valid.sum())
        gt_planes = find_planes(gtd, valid, u, v, rng, tol, minpx, args.max_planes, args.ransac_iters)
        pr_planes = find_planes(aligned, valid, u, v, rng, tol, minpx, args.max_planes, args.ransac_iters)
        if not gt_planes:
            continue

        # --- support agreement: best-matching predicted plane for each GT plane ---
        for mg, _ in gt_planes:
            best = 0.0
            for mp, _ in pr_planes:
                inter = float((mg & mp).sum())
                union = float((mg | mp).sum())
                if union > 0:
                    best = max(best, inter / union)
            ious.append(best)

        # --- orientation error on the SAME support (isolates geometry from detection) ---
        out_c, out_ab, out_all = aligned.copy(), aligned.copy(), aligned.copy()
        for mg, cg in gt_planes:
            cp_ = fit_plane(u, v, aligned, mg)
            # angle between the two disparity-gradient directions
            g1 = np.array([cg[0], cg[1]])
            g2 = np.array([cp_[0], cp_[1]])
            n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
            if n1 > 1e-12 and n2 > 1e-12:
                cosang = float(np.clip(g1 @ g2 / (n1 * n2), -1, 1))
                ang_err.append(float(np.degrees(np.arccos(cosang))))
                rel_ab.append(float(np.linalg.norm(g1 - g2) / max(n1, 1e-12)))
                tilt_mag.append(float(n1 / max(rngv, 1e-12)))
            rel_c.append(float(abs(cg[2] - cp_[2]) / max(rngv, 1e-12)))

            # fix only the offset c
            out_c[mg] = aligned[mg] + (cg[2] - cp_[2])
            # fix only the orientation (a, b), keep the prediction's own offset
            ab_only_gt = cg[0] * u + cg[1] * v + cp_[2]
            ab_only_pr = cp_[0] * u + cp_[1] * v + cp_[2]
            out_ab[mg] = aligned[mg] - ab_only_pr[mg] + ab_only_gt[mg]
            # fix both
            out_all[mg] = aligned[mg] - eval_plane(cp_, u, v)[mg] + eval_plane(cg, u, v)[mg]

        n = int(valid.sum())
        vt = torch.from_numpy(valid)
        gtt = torch.from_numpy(1.0 / np.clip(gtd, 1e-3, None))
        floor = max(1e-3, 0.02 * float(np.median(gtd[valid])))
        for name, arr in [("baseline", aligned), ("fix_c", out_c), ("fix_ab", out_ab), ("fix_all", out_all)]:
            dd = torch.from_numpy(1.0 / np.clip(arr, floor, None))
            acc[name][0] += float(abs_relative_difference(dd, gtt, vt)) * n
            acc[name][1] += n
        nimg += 1

    if nimg == 0:
        raise RuntimeError("No samples scored.")

    base = acc["baseline"][0] / acc["baseline"][1]
    summary = {
        "num_images": nimg,
        "support_iou": {
            "mean": float(np.mean(ious)), "median": float(np.median(ious)),
            "frac_above_0.5": float(np.mean(np.array(ious) > 0.5)),
        },
        "orientation": {
            "angle_deg_mean": float(np.mean(ang_err)), "angle_deg_median": float(np.median(ang_err)),
            "rel_ab_error_median": float(np.median(rel_ab)),
            "rel_c_error_median": float(np.median(rel_c)),
            "gt_tilt_magnitude_median": float(np.median(tilt_mag)),
        },
        "variants": {},
    }
    for kk in variants:
        a = acc[kk][0] / acc[kk][1]
        summary["variants"][kk] = {"abs_rel": a, "gain_pct": 100.0 * (base - a) / base}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    s, o, v = summary["support_iou"], summary["orientation"], summary["variants"]
    print(f"\nPlane diagnosis  n={nimg}  planes matched={len(ious)}")
    print("\n--- does the model know WHERE the planes are? ---")
    print(f"  support IoU (GT plane vs best predicted plane):"
          f"  mean {s['mean']:.3f}   median {s['median']:.3f}")
    print(f"  fraction of GT planes with IoU > 0.5: {s['frac_above_0.5']*100:.0f}%")
    print("\n--- does it get their ORIENTATION right? ---")
    print(f"  gradient-direction error:  mean {o['angle_deg_mean']:.1f} deg   median {o['angle_deg_median']:.1f} deg")
    print(f"  relative error in (a,b):   median {o['rel_ab_error_median']:.3f}")
    print(f"  relative error in c:       median {o['rel_c_error_median']:.4f}")
    print(f"  GT tilt magnitude:         median {o['gt_tilt_magnitude_median']:.3f}")
    print("\n--- which parameter carries the gain? ---")
    print(f"{'variant':<12}{'abs_rel':>10}{'gain':>9}")
    print("-" * 32)
    for kk in variants:
        print(f"{kk:<12}{v[kk]['abs_rel']:>10.5f}{v[kk]['gain_pct']:>8.2f}%")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
