#!/usr/bin/env python
"""Is it the number of sparse points that matters, or the depth range they span?

Marigold-SSD (2026-05) reports a steep drop on out-of-distribution sparsity despite
already randomising condition density during fine-tuning, and names the cause: the sparse
condition is normalised into the VAE's [-1,1] range, so "depth inaccuracies may occur when
depth range of the scene deviates from the range of the depth condition". They list
adaptive normalisation as future work.

Every sparsity study found - including their own Fig. 8 - varies the POINT COUNT. None
separates it from RANGE COVERAGE, yet those are different failure modes with different
fixes: too few points is a sampling problem, too narrow a span is a normalisation problem.
With few points the sampled span is also a poor estimate of the scene's, so the two are
confounded in ordinary uniform sampling.

Measured two ways:
  1. at FIXED point count, does the span the points happen to cover predict the gain?
  2. does deliberately spanning the range (stratified) beat uniform at equal count?

A yes to either makes range coverage an independent factor and points at the
normalisation, not the density. A no closes the direction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
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

Image.MAX_IMAGE_PIXELS = None


def fit_apply_rbf(resid, valid, x, y, sel, smoothing, grid=192):
    """Thin-plate RBF through the sampled residuals, evaluated everywhere.

    Two departures from eval_sparse_points_ceiling's version, both for speed - this
    experiment needs 128 fits per image instead of a handful:

    neighbors=64 there switches scipy to a per-query local solve. With at most 100
    centres the global solve is one small linear system, and is what the local solve
    approximates anyway.

    Evaluating at every valid pixel then costs a 770k x N kernel matrix per fit (0.9 s,
    memory-bound, 7 h over the dataset). A thin-plate spline through <=100 centres has no
    detail to lose at full resolution, so it is evaluated on a coarse grid and bilinearly
    upsampled. Verified against the exact field below (see --check_grid).
    """
    from scipy.interpolate import RBFInterpolator

    pts = np.stack([x[sel], y[sel]], axis=1)
    vals = resid[sel]
    if len(pts) < 4:
        return np.full_like(x, float(vals.mean()) if len(vals) else 0.0)
    try:
        rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                              smoothing=smoothing, degree=1)
    except Exception:
        return np.full_like(x, float(vals.mean()))

    h, w = x.shape
    if grid <= 0 or max(h, w) <= grid:
        out = np.zeros_like(x)
        out[valid] = rbf(np.stack([x[valid], y[valid]], axis=1))
        return out

    gh, gw = max(int(round(h / max(h, w) * grid)), 4), max(int(round(w / max(h, w) * grid)), 4)
    # grid spans the same normalised coords as x, y, so cv2.resize lands back in register
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), gw),
                         np.linspace(y.min(), y.max(), gh))
    coarse = rbf(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(gh, gw)
    return cv2.resize(coarse.astype(np.float64), (w, h), interpolation=cv2.INTER_LINEAR)


def parse_args():
    p = argparse.ArgumentParser(description="Separate sparse-point density from range coverage.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--processing_res", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="output/eval_sparse_range")
    p.add_argument("--illum", type=str, default="all")
    p.add_argument("--eval_scale", type=int, default=4)
    p.add_argument("--n_points", type=int, nargs="+", default=[10, 20, 50, 100])
    p.add_argument("--draws", type=int, default=8, help="Random draws per image per setting.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rbf_grid", type=int, default=192,
                   help="Long edge of the grid the RBF is evaluated on (0 = every pixel).")
    p.add_argument("--check_grid", action="store_true",
                   help="On the first image, report the coarse-grid vs exact discrepancy.")
    return p.parse_args()


def sample_uniform(rng, vidx, n):
    return rng.choice(vidx, size=min(n, vidx.size), replace=False)


def sample_stratified(rng, vidx, n, depth_flat):
    """One point per equal-count depth bin: maximum range coverage at the same count."""
    order = vidx[np.argsort(depth_flat[vidx])]
    if order.size <= n:
        return order
    edges = np.linspace(0, order.size, n + 1).astype(int)
    return np.array([order[rng.integers(edges[i], max(edges[i + 1], edges[i] + 1))]
                     for i in range(n)])


def sample_spatial_half(rng, vidx, n, shape):
    """CONTROL: confined to half the IMAGE, depth unrestricted.

    narrow_near / narrow_far cut on depth, but near pixels cluster spatially (floor,
    foreground), so those conditions also force the interpolator to extrapolate across
    empty image regions. If this control - spatially confined, depth-range free - breaks
    just as badly, the damage is spatial extrapolation and says nothing about the depth
    range or its normalisation.
    """
    h, w = shape
    yy, xx = np.divmod(vidx, w)
    axis, side = rng.integers(2), rng.integers(2)
    c = (yy < h / 2) if axis == 0 else (xx < w / 2)
    sel = vidx[c if side == 0 else ~c]
    if sel.size < n:
        sel = vidx
    return rng.choice(sel, size=min(n, sel.size), replace=False)


def spatial_spread(pick, shape):
    """Fraction of the image the samples' bounding box covers - the confound, quantified."""
    h, w = shape
    yy, xx = np.divmod(pick, w)
    return float((np.ptp(yy) / max(h - 1, 1)) * (np.ptp(xx) / max(w - 1, 1)))


def sample_narrow(rng, vidx, n, depth_flat, half="near"):
    """Points confined to one half of the depth range: deliberately poor coverage."""
    d = depth_flat[vidx]
    mid = np.median(d)
    sel = vidx[d <= mid] if half == "near" else vidx[d > mid]
    if sel.size < n:
        sel = vidx
    return rng.choice(sel, size=min(n, sel.size), replace=False)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.data_root)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = ["uniform", "stratified", "narrow_near", "narrow_far", "spatial_half"]
    # err, depth coverage, spatial spread, weight
    acc = {(s, n): [0.0, 0.0, 0.0, 0] for s in strategies for n in args.n_points}
    base_acc = [0.0, 0]
    checked = False
    # per-draw records at fixed N, for the coverage-vs-gain correlation
    pairs = {n: [] for n in args.n_points}

    for scene, frame, _ in tqdm(list_samples(root, args.illum), desc="range_vs_density"):
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
        vv, uu = np.mgrid[0:h, 0:w]
        x = (uu / w - 0.5).astype(np.float64)
        y = (vv / h - 0.5).astype(np.float64)

        n_valid = int(valid.sum())
        vt = torch.from_numpy(valid)
        gtt = torch.from_numpy(1.0 / np.clip(gtd, 1e-3, None))
        floor = max(1e-3, 0.02 * float(np.median(gtd[valid])))

        def score(field):
            d = aligned - field
            return float(abs_relative_difference(
                torch.from_numpy(1.0 / np.clip(d, floor, None)), gtt, vt))

        b = score(np.zeros_like(aligned))
        base_acc[0] += b * n_valid
        base_acc[1] += n_valid

        if args.check_grid and not checked:
            checked = True
            pk = sample_uniform(rng, np.flatnonzero(valid.ravel()), max(args.n_points))
            s0 = np.zeros(valid.size, bool); s0[pk] = True; s0 = s0.reshape(valid.shape)
            ex = score(fit_apply_rbf(resid, valid, x, y, s0, 1e-3, 0))
            ap = score(fit_apply_rbf(resid, valid, x, y, s0, 1e-3, args.rbf_grid))
            print(f"[grid check] exact abs_rel={ex:.6f}  grid{args.rbf_grid} abs_rel={ap:.6f}  "
                  f"diff={abs(ex-ap)/ex*100:.4f}% of value", flush=True)

        vidx = np.flatnonzero(valid.ravel())
        gflat = gtd.ravel()
        # the scene's own span, as the reference for coverage
        lo, hi = np.percentile(gflat[vidx], [1, 99])
        span = max(hi - lo, 1e-9)

        for n in args.n_points:
            for s in strategies:
                for _ in range(args.draws):
                    if s == "uniform":
                        pick = sample_uniform(rng, vidx, n)
                    elif s == "stratified":
                        pick = sample_stratified(rng, vidx, n, gflat)
                    elif s == "spatial_half":
                        pick = sample_spatial_half(rng, vidx, n, valid.shape)
                    else:
                        pick = sample_narrow(rng, vidx, n, gflat,
                                             "near" if s == "narrow_near" else "far")
                    sel = np.zeros(valid.size, dtype=bool)
                    sel[pick] = True
                    sel = sel.reshape(valid.shape)
                    cov = float((gflat[pick].max() - gflat[pick].min()) / span)
                    g = score(fit_apply_rbf(resid, valid, x, y, sel, 1e-3, args.rbf_grid))
                    e = acc[(s, n)]
                    e[0] += g * n_valid
                    e[1] += cov * n_valid
                    e[2] += spatial_spread(pick, valid.shape) * n_valid
                    e[3] += n_valid
                    if s == "uniform":
                        pairs[n].append((cov, 100.0 * (b - g) / max(b, 1e-12)))

    base = base_acc[0] / base_acc[1]
    summary = {"baseline_abs_rel": base, "draws": args.draws, "settings": {}, "corr_cov_gain": {}}
    for (s, n), e in acc.items():
        if e[3] == 0:
            continue
        a = e[0] / e[3]
        summary["settings"][f"{s}_n{n}"] = {
            "abs_rel": a, "gain_pct": 100.0 * (base - a) / base,
            "coverage": e[1] / e[3], "spatial_spread": e[2] / e[3],
        }
    for n, pl in pairs.items():
        if len(pl) > 10:
            c, g = np.array(pl).T
            summary["corr_cov_gain"][str(n)] = {
                "corr": float(np.corrcoef(c, g)[0, 1]), "n_draws": len(pl),
                "coverage_mean": float(c.mean()), "coverage_std": float(c.std()),
            }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    S = summary["settings"]
    print(f"\nRange vs density  baseline abs_rel={base:.5f}  draws={args.draws}")
    print("\n--- gain by strategy (same point count, different range coverage) ---")
    print(f"{'N':>5}" + "".join(f"{s:>22}" for s in strategies))
    print("-" * (5 + 22 * len(strategies)))
    for n in args.n_points:
        row = f"{n:>5}"
        for s in strategies:
            k = f"{s}_n{n}"
            row += (f"{S[k]['gain_pct']:>9.1f}% d{S[k]['coverage']:.2f} s{S[k]['spatial_spread']:.2f}"
                    if k in S else f"{'-':>22}")
        print(row)
    print("   d = fraction of the scene's depth span covered; s = fraction of image area spanned")
    print("\n--- at FIXED point count, does coverage predict the gain? ---")
    print(f"{'N':>5}{'corr(coverage, gain)':>24}{'coverage mean+-sd':>22}")
    print("-" * 52)
    for n in args.n_points:
        c = summary["corr_cov_gain"].get(str(n))
        if c:
            print(f"{n:>5}{c['corr']:>24.3f}{c['coverage_mean']:>15.2f} +-{c['coverage_std']:.2f}")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
