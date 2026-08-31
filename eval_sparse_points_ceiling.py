#!/usr/bin/env python
"""How many ground-truth depth points does it take to recover the global error?

Post-processing on the model's own outputs cannot reach the 22.8% tilt oracle: the tilt
is not a fixed bias, it survives averaging because every perturbation makes the same
structural mistake, and it is not predictable from GT-free features (R^2 < 0). What is
left is adding information the model does not have. Sparse depth is the cheapest such
information - a few points from LiDAR, SfM or stereo - and the error being 75%
low-frequency is exactly the regime where a handful of points should pin it down.

Measured before building anything, because skipping that step is what produced five
dead ends in this project.

Design note: the standard protocol already scale/shift-aligns the prediction using ALL
valid GT pixels, so replacing that with sparse points would confound "sparse alignment
is noisy" with the question being asked. Instead the aligned prediction is the starting
point, and the sparse points are used only to fit a correction to what remains. That
makes the numbers directly comparable to the all-pixel oracles.

Correction families, fitted from N points only:
  tilt   a*x + b*y + c          (3 params)  - the 22.8% oracle's form
  quad   full quadratic         (6 params)  - the 36.4% oracle's form
  rbf    thin-plate spline      (smooth)    - approaches the low-frequency oracle
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
    p = argparse.ArgumentParser(description="Sparse-point ceiling for the global error.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--processing_res", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="output/eval_sparse_points")
    p.add_argument("--illum", type=str, default="all")
    p.add_argument("--eval_scale", type=int, default=4)
    p.add_argument("--n_points", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50, 100, 500])
    p.add_argument("--draws", type=int, default=5, help="Random point sets averaged per image.")
    p.add_argument("--ridge", type=float, default=1e-6, help="Ridge term, so few-point fits stay finite.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def basis(x, y, kind):
    if kind == "tilt":
        return [np.ones_like(x), x, y]
    if kind == "quad":
        return [np.ones_like(x), x, y, x * x, x * y, y * y]
    raise ValueError(kind)


def fit_apply(resid, valid, x, y, sel, kind, ridge):
    """Fit a polynomial to the residual at `sel` only, evaluate it everywhere."""
    B = basis(x, y, kind)
    A = np.stack([b[sel] for b in B], axis=1)
    t = resid[sel]
    # ridge keeps the system solvable when there are fewer points than parameters
    G = A.T @ A + ridge * np.eye(A.shape[1])
    coef = np.linalg.solve(G, A.T @ t)
    return sum(c * b for c, b in zip(coef, B))


def fit_apply_rbf(resid, valid, x, y, sel, smoothing):
    from scipy.interpolate import RBFInterpolator

    pts = np.stack([x[sel], y[sel]], axis=1)
    vals = resid[sel]
    if len(pts) < 3:
        return np.full_like(x, float(vals.mean()) if len(vals) else 0.0)
    try:
        rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=smoothing,
                             degree=1, neighbors=min(64, len(pts)))
    except Exception:
        return np.full_like(x, float(vals.mean()))
    q = np.stack([x[valid], y[valid]], axis=1)
    out = np.zeros_like(x)
    out[valid] = rbf(q)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.data_root)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = ["tilt", "quad", "rbf"]
    keys = ["baseline"] + [f"{f}_n{n}" for n in args.n_points for f in families] \
           + ["oracle_tilt", "oracle_quad"]
    acc = {k: [0.0, 0.0, 0] for k in keys}

    for scene, frame, _ in tqdm(list_samples(root, args.illum), desc="sparse_points"):
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
        resid = aligned - gtd
        h, w = aligned.shape
        yy, xx = np.mgrid[0:h, 0:w]
        x = (xx / w - 0.5).astype(np.float64)
        y = (yy / h - 0.5).astype(np.float64)

        n = int(valid.sum())
        vt = torch.from_numpy(valid)
        gtt = torch.from_numpy(1.0 / np.clip(gtd, 1e-3, None))
        floor = max(1e-3, 0.02 * float(np.median(gtd[valid])))

        def record(name, field):
            d = aligned - field
            dd = torch.from_numpy(1.0 / np.clip(d, floor, None))
            e = acc[name]
            e[0] += float(abs_relative_difference(dd, gtt, vt)) * n
            e[1] += float(delta1_acc(dd, gtt, vt)) * n
            e[2] += n

        record("baseline", np.zeros_like(aligned))
        record("oracle_tilt", fit_apply(resid, valid, x, y, valid, "tilt", args.ridge))
        record("oracle_quad", fit_apply(resid, valid, x, y, valid, "quad", args.ridge))

        vidx = np.flatnonzero(valid.ravel())
        for npts in args.n_points:
            sums = {f: np.zeros_like(aligned) for f in families}
            for _ in range(args.draws):
                pick = rng.choice(vidx, size=min(npts, vidx.size), replace=False)
                sel = np.zeros(valid.size, dtype=bool)
                sel[pick] = True
                sel = sel.reshape(valid.shape)
                sums["tilt"] += fit_apply(resid, valid, x, y, sel, "tilt", args.ridge)
                sums["quad"] += fit_apply(resid, valid, x, y, sel, "quad", args.ridge)
                sums["rbf"] += fit_apply_rbf(resid, valid, x, y, sel, smoothing=1e-3)
            for f in families:
                record(f"{f}_n{npts}", sums[f] / args.draws)

    base = acc["baseline"][0] / acc["baseline"][2]
    summary = {
        "processing_res": args.processing_res,
        "draws": args.draws,
        "n_points": args.n_points,
        "variants": {},
    }
    for kk in keys:
        if acc[kk][2] == 0:
            continue
        a = acc[kk][0] / acc[kk][2]
        summary["variants"][kk] = {
            "abs_rel": a, "delta1": acc[kk][1] / acc[kk][2],
            "gain_pct": 100.0 * (base - a) / base,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    v = summary["variants"]
    print(f"\nSparse-point ceiling  res={args.processing_res}  draws={args.draws}")
    print(f"baseline abs_rel = {base:.5f}")
    print(f"all-pixel oracles:  tilt {v['oracle_tilt']['gain_pct']:+.1f}%"
          f"   quad {v['oracle_quad']['gain_pct']:+.1f}%\n")
    print(f"{'N points':>9}" + "".join(f"{f:>12}" for f in families))
    print("-" * 46)
    for npts in args.n_points:
        row = f"{npts:>9}"
        for f in families:
            kk = f"{f}_n{npts}"
            row += f"{v[kk]['gain_pct']:>11.1f}%" if kk in v else f"{'-':>12}"
        print(row)
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
