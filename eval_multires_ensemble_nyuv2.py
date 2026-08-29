#!/usr/bin/env python
"""Does the multi-resolution ensemble transfer from Booster to NYUv2?

On Booster it beat the best single resolution by 5.3% with no training, and a blur
control ruled out "the average is just smoother". Booster is 12 Mpx close-ups; NYUv2 is
0.3 Mpx wide room shots with a different optimum (768 vs 512), so transfer is not
implied - if the gain is a property of the model rather than of that dataset, it should
appear here too.

Same protocol: align each prediction to a REFERENCE PREDICTION, average, and touch GT
only to score. Includes the blur control, since that is what made the Booster result
trustworthy.
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

from evaluation.util.alignment import align_depth_least_square, depth2disparity
from evaluation.util.metric import abs_relative_difference, delta1_acc
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from eval_booster_multires_ensemble import fit_to_reference


def parse_args():
    p = argparse.ArgumentParser(description="Multi-resolution ensemble on NYUv2.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--resolutions", type=int, nargs="+", default=[512, 640, 768])
    p.add_argument("--reference", type=int, default=768)
    p.add_argument("--output_dir", type=str, default="output/eval_nyuv2_multires")
    p.add_argument("--blur_control", type=float, nargs="+", default=[2, 4, 8])
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = list(args.resolutions)

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    names = [f"single_{r}" for r in res] + ["ensemble"] + [f"blur_s{s:g}" for s in args.blur_control]
    acc = {k: [0.0, 0.0, 0] for k in names}
    corr_sum = np.zeros((len(res), len(res)))
    corr_n = 0

    import cv2

    for rgb_path, depth_path in tqdm(pairs, desc="nyuv2_multires"):
        rel = rgb_path.relative_to(rgb_dir)
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape

        preds, missing = {}, False
        for r in res:
            p = Path(args.pred_cache_dir) / f"res{r}" / rel.parent / f"{rel.stem}_pred.npy"
            if not p.is_file():
                missing = True
                break
            a = np.load(p).astype(np.float32)
            if a.shape != (h, w):
                a = np.array(Image.fromarray(a).resize((w, h), Image.BILINEAR), dtype=np.float32)
            preds[r] = a
        if missing:
            continue

        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        gt_disp, gt_nn = depth2disparity(depth=gt, return_mask=True)
        valid &= gt_nn
        for r in res:
            valid &= np.isfinite(preds[r]) & (preds[r] > 0)
        if valid.sum() < 1000:
            continue

        ref = preds[args.reference]
        ens = np.mean([fit_to_reference(preds[r], ref, valid) for r in res], axis=0)

        cands = {f"single_{r}": preds[r].astype(np.float64) for r in res}
        cands["ensemble"] = ens
        for s in args.blur_control:
            ks = int(2 * round(3 * s) + 1)
            cands[f"blur_s{s:g}"] = cv2.GaussianBlur(
                ref.astype(np.float32), (ks, ks), s, borderType=cv2.BORDER_REPLICATE
            ).astype(np.float64)

        n = int(valid.sum())
        vt, gt_t = torch.from_numpy(valid), torch.from_numpy(gt)
        floor = max(1e-3, 0.02 * float(np.median(gt_disp[valid])))
        resid = []
        for name, arr in cands.items():
            al = align_depth_least_square(
                gt_arr=gt_disp, pred_arr=arr, valid_mask_arr=valid, return_scale_shift=False
            )
            depth = np.clip(1.0 / np.clip(al, floor, None), 1e-3, 10.0)
            dt = torch.from_numpy(depth)
            e = acc[name]
            e[0] += float(abs_relative_difference(dt, gt_t, vt)) * n
            e[1] += float(delta1_acc(dt, gt_t, vt)) * n
            e[2] += n
            if name.startswith("single_"):
                resid.append((al - gt_disp)[valid])

        R = np.corrcoef(np.stack(resid))
        if np.all(np.isfinite(R)):
            corr_sum += R
            corr_n += 1

    best_single = min(acc[f"single_{r}"][0] / acc[f"single_{r}"][2] for r in res)
    ens_a = acc["ensemble"][0] / acc["ensemble"][2]
    summary = {
        "resolutions": res,
        "reference": args.reference,
        "variants": {k: {"abs_rel": v[0] / v[2], "delta1": v[1] / v[2]} for k, v in acc.items()},
        "gain_vs_best_single_pct": 100.0 * (best_single - ens_a) / best_single,
        "residual_correlation": (corr_sum / max(corr_n, 1)).tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nNYUv2 multi-resolution ensemble  ref={args.reference}  res={res}")
    print(f"{'variant':<16}{'abs_rel':>10}{'delta1':>9}{'gain':>9}")
    print("-" * 45)
    for k in names:
        v = summary["variants"][k]
        print(f"{k:<16}{v['abs_rel']:>10.5f}{v['delta1']:>9.4f}"
              f"{100*(best_single - v['abs_rel'])/best_single:>8.2f}%")
    R = np.array(summary["residual_correlation"])
    off = R[np.triu_indices(len(res), 1)]
    print(f"\nmean residual correlation between resolutions: {off.mean():.3f}")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
