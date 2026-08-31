#!/usr/bin/env python
"""Do resolution and flip perturbations combine, or overlap?

They behave oppositely on the two datasets:

                    Booster   NYUv2
  multi-resolution   +7.29%   +0.84%   (fails on NYUv2)
  horizontal flip    +4.63%   +2.60%

If the two cancel different parts of the error, combining them should beat either alone.
If they cancel the same part, the combination lands near the better of the two. NYUv2 is
the sharper test: multi-resolution alone does essentially nothing there, so any gain
beyond the flip's +2.60% has to come from the interaction.

TTA members are aligned to a REFERENCE PREDICTION, never to GT; GT is touched once to
score. Reports the residual correlation between the two perturbation types, which says
directly whether they are redundant.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from itertools import product
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
from pipeline import LotusDPipeline
from utils.seed_all import seed_all


def parse_args():
    p = argparse.ArgumentParser(description="Combined resolution+flip TTA on NYUv2.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--flip_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred_flip")
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--resolutions", type=int, nargs="+", default=[512, 640, 768])
    p.add_argument("--reference", type=int, default=768)
    p.add_argument("--output_dir", type=str, default="output/eval_nyuv2_tta_combined")
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def predict(pipe, rgb_np, processing_res, generator):
    device = pipe.device
    im = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    im = (im / 127.5 - 1.0).to(device)
    te = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    te = torch.cat([torch.sin(te), torch.cos(te)], dim=-1)
    ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device_type=device.type)
    with ctx:
        o = pipe(
            rgb_in=im, prompt="", num_inference_steps=1, generator=generator, output_type="np",
            timesteps=[999], task_emb=te, processing_res=processing_res, match_input_res=True,
        ).images[0]
    return (o.mean(axis=-1) if o.ndim == 3 else o).astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    seed_all(args.seed)
    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = list(args.resolutions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half_precision else torch.float32
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipe = None

    def get_pipe():
        nonlocal pipe
        if pipe is None:
            logging.info("Loading Lotus pipeline: %s", args.core_model)
            pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=dtype).to(device)
            pipe.set_progress_bar_config(disable=True)
        return pipe

    def load_or_build(rel, r, flip, rgb_getter, shape):
        root = Path(args.flip_cache_dir if flip else args.cache_dir) / f"res{r}"
        path = root / rel.parent / f"{rel.stem}_pred.npy"
        if path.is_file():
            a = np.load(path).astype(np.float32)
        else:
            rgb = rgb_getter()
            src = rgb[:, ::-1].copy() if flip else rgb
            a = predict(get_pipe(), src, r, generator)
            if flip:
                a = a[:, ::-1].copy()
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, a.astype(np.float16))
        if a.shape != shape:
            a = np.array(Image.fromarray(a).resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)
        return a

    names = ["single", "multires", "flip_tta", "combined"]
    acc = {k: [0.0, 0.0, 0] for k in names}
    wins = {k: 0 for k in names}
    corr_rs, corr_fl = [], []
    n_img = 0

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    for rgb_path, depth_path in tqdm(pairs, desc="tta_combined"):
        rel = rgb_path.relative_to(rgb_dir)
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        _rgb = {}

        def rgb_getter():
            if "v" not in _rgb:
                _rgb["v"] = np.array(Image.open(rgb_path).convert("RGB"))
            return _rgb["v"]

        P = {(r, f): load_or_build(rel, r, f, rgb_getter, (h, w)) for r, f in product(res, (False, True))}

        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        gt_disp, nn = depth2disparity(depth=gt, return_mask=True)
        valid &= nn
        for a in P.values():
            valid &= np.isfinite(a) & (a > 0)
        if valid.sum() < 1000:
            continue

        ref = P[(args.reference, False)]
        A = {k: fit_to_reference(v, ref, valid) for k, v in P.items()}
        groups = {
            "single": [A[(args.reference, False)]],
            "multires": [A[(r, False)] for r in res],
            "flip_tta": [A[(args.reference, False)], A[(args.reference, True)]],
            "combined": list(A.values()),
        }

        n = int(valid.sum())
        vt, gt_t = torch.from_numpy(valid), torch.from_numpy(gt)
        floor = max(1e-3, 0.02 * float(np.median(gt_disp[valid])))
        sc = {}
        for name, members in groups.items():
            al = align_depth_least_square(
                gt_arr=gt_disp, pred_arr=np.mean(members, axis=0),
                valid_mask_arr=valid, return_scale_shift=False,
            )
            d = np.clip(1.0 / np.clip(al, floor, None), 1e-3, 10.0)
            dt = torch.from_numpy(d)
            a = float(abs_relative_difference(dt, gt_t, vt))
            acc[name][0] += a * n
            acc[name][1] += float(delta1_acc(dt, gt_t, vt)) * n
            acc[name][2] += n
            sc[name] = a
        for k in names:
            if sc[k] <= sc["single"]:
                wins[k] += 1
        n_img += 1

        # Are the two perturbations redundant? Compare each one's residual against
        # the reference prediction's.
        def res_of(arr):
            al = align_depth_least_square(
                gt_arr=gt_disp, pred_arr=arr, valid_mask_arr=valid, return_scale_shift=False
            )
            return (al - gt_disp)[valid]

        r0 = res_of(A[(args.reference, False)])
        c1 = np.corrcoef(r0, res_of(A[(res[0], False)]))[0, 1]
        c2 = np.corrcoef(r0, res_of(A[(args.reference, True)]))[0, 1]
        if np.isfinite(c1):
            corr_rs.append(c1)
        if np.isfinite(c2):
            corr_fl.append(c2)

    base = acc["single"][0] / acc["single"][2]
    summary = {
        "resolutions": res,
        "reference": args.reference,
        "num_images": n_img,
        "variants": {
            k: {
                "abs_rel": v[0] / v[2],
                "delta1": v[1] / v[2],
                "gain_pct": 100.0 * (base - v[0] / v[2]) / base,
                "win_rate": wins[k] / max(n_img, 1),
            }
            for k, v in acc.items()
        },
        "residual_correlation": {
            "resolution_change": float(np.mean(corr_rs)),
            "flip": float(np.mean(corr_fl)),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nNYUv2 combined TTA  res={res}  ref={args.reference}  n={n_img}")
    print(f"{'variant':<12}{'members':>9}{'abs_rel':>10}{'delta1':>9}{'gain':>9}{'win':>8}")
    print("-" * 58)
    sizes = {"single": 1, "multires": len(res), "flip_tta": 2, "combined": 2 * len(res)}
    for k in names:
        v = summary["variants"][k]
        print(f"{k:<12}{sizes[k]:>9}{v['abs_rel']:>10.5f}{v['delta1']:>9.4f}"
              f"{v['gain_pct']:>8.2f}%{v['win_rate']*100:>7.0f}%")
    c = summary["residual_correlation"]
    print(f"\nresidual correlation vs the reference prediction:")
    print(f"  resolution change  {c['resolution_change']:.3f}")
    print(f"  horizontal flip    {c['flip']:.3f}")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
