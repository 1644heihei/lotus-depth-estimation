#!/usr/bin/env python
"""Does averaging Lotus across resolutions cancel its global error?

The decomposition showed the error is mostly low-frequency and, per the tilt analysis,
differs image to image rather than being a fixed bias. If each resolution's global error
is at least partly independent, averaging several should cancel some of it - a
training-free gain, unlike everything this project tried before.

Kept honest: predictions are aligned to a REFERENCE PREDICTION, never to GT, before
averaging. GT is touched once at the end to score. Aligning each to GT first and then
averaging would leak the answer in and manufacture a gain.

Also reports the correlation between resolutions' residuals, which decides the question
directly: highly correlated residuals cannot cancel, however many you average.
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
    p = argparse.ArgumentParser(description="Multi-resolution ensemble on Booster.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--resolutions", type=int, nargs="+", default=[384, 512, 640, 768])
    p.add_argument("--reference", type=int, default=512, help="Resolution used as the alignment anchor.")
    p.add_argument("--output_dir", type=str, default="output/eval_booster_multires")
    p.add_argument("--illum", type=str, default="im0")
    p.add_argument("--eval_scale", type=int, default=2)
    p.add_argument("--max_scenes", type=int, default=0)
    return p.parse_args()


def fit_to_reference(src, ref, mask):
    """Affine-fit src onto ref over mask. Uses no ground truth."""
    x, y = src[mask].astype(np.float64), ref[mask].astype(np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 64 or np.var(x) < 1e-12:
        return src.astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if not np.isfinite(a) or a <= 1e-8:
        return src.astype(np.float64)
    return a * src.astype(np.float64) + b


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(root, args.illum)
    if args.max_scenes > 0:
        keep = set(sorted({s.name for s, _, _ in samples})[: args.max_scenes])
        samples = [t for t in samples if t[0].name in keep]

    res = list(args.resolutions)
    variants = [f"single_{r}" for r in res] + ["ensemble"]
    acc = {v: {"err": 0.0, "d1": 0.0, "n": 0} for v in variants}
    corr_sum = np.zeros((len(res), len(res)))
    corr_n = 0

    for scene, frame, _ in tqdm(samples, desc="multires"):
        k = args.eval_scale
        preds = {}
        missing = False
        for r in res:
            p = Path(args.pred_cache_dir) / f"res{r}" / scene.name / f"{frame}_pred.npy"
            if not p.is_file():
                missing = True
                break
            preds[r] = downsample(np.load(p).astype(np.float32), k)
        if missing:
            continue

        gt_disp = downsample(np.load(scene / "disp_00.npy").astype(np.float32), k)
        occ = downsample(np.array(Image.open(scene / "mask_00.png")), k, nearest=True)
        valid = np.isfinite(gt_disp) & (gt_disp > 0) & (occ > 127)
        for r in res:
            valid &= np.isfinite(preds[r]) & (preds[r] > 0)
        if valid.sum() < 1000:
            continue

        ref = preds[args.reference]
        stack = [fit_to_reference(preds[r], ref, valid) for r in res]
        ens = np.mean(stack, axis=0)

        gt_d = gt_disp.astype(np.float64)
        n = int(valid.sum())
        vt = torch.from_numpy(valid)
        gt_depth = torch.from_numpy(1.0 / np.clip(gt_d, 1e-3, None))
        floor = max(1e-3, 0.02 * float(np.median(gt_d[valid])))

        resid = []
        for name, arr in [(f"single_{r}", preds[r]) for r in res] + [("ensemble", ens)]:
            al = align_depth_least_square(
                gt_arr=gt_d, pred_arr=np.asarray(arr, dtype=np.float64),
                valid_mask_arr=valid, return_scale_shift=False,
            )
            dd = torch.from_numpy(1.0 / np.clip(al, floor, None))
            acc[name]["err"] += float(abs_relative_difference(dd, gt_depth, vt)) * n
            acc[name]["d1"] += float(delta1_acc(dd, gt_depth, vt)) * n
            acc[name]["n"] += n
            if name != "ensemble":
                resid.append((al - gt_d)[valid])

        R = np.corrcoef(np.stack(resid))
        if np.all(np.isfinite(R)):
            corr_sum += R
            corr_n += 1

    if acc["ensemble"]["n"] == 0:
        raise RuntimeError("No samples scored (missing caches for some resolutions?).")

    summary = {"resolutions": res, "reference": args.reference, "variants": {}}
    for v in variants:
        summary["variants"][v] = {
            "abs_rel": acc[v]["err"] / acc[v]["n"],
            "delta1": acc[v]["d1"] / acc[v]["n"],
        }
    best_single = min(summary["variants"][f"single_{r}"]["abs_rel"] for r in res)
    ens_absrel = summary["variants"]["ensemble"]["abs_rel"]
    summary["gain_vs_best_single_pct"] = 100.0 * (best_single - ens_absrel) / best_single
    summary["residual_correlation"] = (corr_sum / max(corr_n, 1)).tolist()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBooster multi-resolution ensemble  ref={args.reference}  n={acc['ensemble']['n']//1}px-weighted")
    print(f"{'variant':<16}{'abs_rel':>10}{'delta1':>9}")
    print("-" * 36)
    for r in res:
        s = summary["variants"][f"single_{r}"]
        print(f"{'single ' + str(r):<16}{s['abs_rel']:>10.5f}{s['delta1']:>9.4f}")
    s = summary["variants"]["ensemble"]
    print(f"{'ENSEMBLE':<16}{s['abs_rel']:>10.5f}{s['delta1']:>9.4f}")
    print(f"\nvs best single: {summary['gain_vs_best_single_pct']:+.2f}%")
    print("\nresidual correlation between resolutions (1.0 = identical error, cannot cancel):")
    R = np.array(summary["residual_correlation"])
    print("        " + "".join(f"{r:>8}" for r in res))
    for i, r in enumerate(res):
        print(f"{r:>8}" + "".join(f"{R[i, j]:>8.3f}" for j in range(len(res))))
    off = R[np.triu_indices(len(res), 1)]
    print(f"\nmean off-diagonal correlation: {off.mean():.3f}")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
