#!/usr/bin/env python
"""Does horizontal-flip TTA decorrelate the GLOBAL error, where resolution does not?

Decomposing the multi-resolution ensemble showed it reduces the local component by 21.5%
but the global one by only 3.7% - averaging cancels the high-frequency noise that differs
between resolutions, while the scene-level structure error is correlated enough (0.67) to
survive. The 75% of error that lives in the low-frequency band is therefore untouched.

Changing resolution rescales the image but preserves its geometry, so the model makes
much the same structural mistake each time. A horizontal flip changes the geometry
outright - vanishing points, ground-plane cues and object layout all move - so if any
cheap perturbation decorrelates the global component, this is the candidate.

Reports the residual correlation split into its low- and high-frequency parts, which is
what actually answers the question: a lower global correlation than resolution gives is
the result that matters, not the headline abs_rel.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
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
from eval_booster_multires_ensemble import fit_to_reference
from pipeline import LotusDPipeline
from utils.seed_all import seed_all

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    p = argparse.ArgumentParser(description="Horizontal-flip TTA on Booster.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--flip_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache_flip")
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--processing_res", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="output/eval_booster_flip_tta")
    p.add_argument("--illum", type=str, default="all")
    p.add_argument("--eval_scale", type=int, default=2)
    p.add_argument("--sigma", type=float, default=64.0, help="Split frequency for the correlation analysis.")
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def predict(pipe, rgb_np, timestep, processing_res, generator):
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = (image / 127.5 - 1.0).to(device)
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device_type=device.type)
    with ctx:
        out = pipe(
            rgb_in=image, prompt="", num_inference_steps=1, generator=generator,
            output_type="np", timesteps=[timestep], task_emb=task_emb,
            processing_res=processing_res, match_input_res=True,
        ).images[0]
    return (out.mean(axis=-1) if out.ndim == 3 else out).astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    seed_all(args.seed)
    import cv2

    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    fcache = Path(args.flip_cache_dir) / f"res{args.processing_res}"

    samples = list_samples(root, args.illum)
    if args.max_scenes > 0:
        keep = set(sorted({s.name for s, _, _ in samples})[: args.max_scenes])
        samples = [t for t in samples if t[0].name in keep]

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

    names = ["single", "flip_only", "flip_tta"]
    acc = {k: [0.0, 0.0, 0] for k in names}
    corr_all, corr_lo, corr_hi = [], [], []

    for scene, frame, rgb_path in tqdm(samples, desc="flip_tta"):
        cpath = cache / scene.name / f"{frame}_pred.npy"
        if not cpath.is_file():
            continue
        fpath = fcache / scene.name / f"{frame}_pred.npy"
        if fpath.is_file():
            flip_pred = np.load(fpath).astype(np.float32)
        else:
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            p = predict(get_pipe(), rgb[:, ::-1].copy(), args.timestep, args.processing_res, generator)
            flip_pred = p[:, ::-1].copy()  # undo the flip so it lines up with the original
            fpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(fpath, flip_pred.astype(np.float16))

        k = args.eval_scale
        base = downsample(np.load(cpath).astype(np.float32), k)
        flip = downsample(flip_pred, k)
        gt = downsample(np.load(scene / "disp_00.npy").astype(np.float32), k)
        occ = downsample(np.array(Image.open(scene / "mask_00.png")), k, nearest=True)

        valid = (np.isfinite(gt) & (gt > 0) & (occ > 127)
                 & np.isfinite(base) & (base > 0) & np.isfinite(flip) & (flip > 0))
        if valid.sum() < 1000:
            continue

        flip_a = fit_to_reference(flip, base, valid)
        tta = 0.5 * (base.astype(np.float64) + flip_a)

        gtd = gt.astype(np.float64)
        n = int(valid.sum())
        vt = torch.from_numpy(valid)
        gtt = torch.from_numpy(1.0 / np.clip(gtd, 1e-3, None))
        floor = max(1e-3, 0.02 * float(np.median(gtd[valid])))

        resid = {}
        for name, arr in [("single", base.astype(np.float64)), ("flip_only", flip_a), ("flip_tta", tta)]:
            al = align_depth_least_square(
                gt_arr=gtd, pred_arr=arr, valid_mask_arr=valid, return_scale_shift=False
            )
            dd = torch.from_numpy(1.0 / np.clip(al, floor, None))
            e = acc[name]
            e[0] += float(abs_relative_difference(dd, gtt, vt)) * n
            e[1] += float(delta1_acc(dd, gtt, vt)) * n
            e[2] += n
            if name in ("single", "flip_only"):
                resid[name] = al - gtd

        # Split each residual into low and high frequency, then correlate separately.
        # The headline question is whether the GLOBAL parts decorrelate.
        ks = int(2 * round(3 * args.sigma) + 1)
        wgt = valid.astype(np.float32)
        wb = cv2.GaussianBlur(wgt, (ks, ks), args.sigma, borderType=cv2.BORDER_REPLICATE)
        lo, hi = {}, {}
        for kk, r in resid.items():
            rb = cv2.GaussianBlur(np.where(valid, r, 0.0).astype(np.float32), (ks, ks),
                                  args.sigma, borderType=cv2.BORDER_REPLICATE)
            lo[kk] = np.where(wb > 1e-3, rb / np.maximum(wb, 1e-3), 0.0)
            hi[kk] = r - lo[kk]
        for store, d in ((corr_all, resid), (corr_lo, lo), (corr_hi, hi)):
            c = np.corrcoef(d["single"][valid], d["flip_only"][valid])[0, 1]
            if np.isfinite(c):
                store.append(c)

    if acc["flip_tta"][2] == 0:
        raise RuntimeError("No samples scored.")

    summary = {
        "processing_res": args.processing_res,
        "variants": {k: {"abs_rel": v[0] / v[2], "delta1": v[1] / v[2]} for k, v in acc.items()},
        "residual_correlation": {
            "all": float(np.mean(corr_all)),
            "low_freq": float(np.mean(corr_lo)),
            "high_freq": float(np.mean(corr_hi)),
        },
    }
    s = summary["variants"]
    summary["gain_pct"] = 100.0 * (s["single"]["abs_rel"] - s["flip_tta"]["abs_rel"]) / s["single"]["abs_rel"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBooster flip TTA  res={args.processing_res}  n={len(corr_all)}")
    print(f"{'variant':<14}{'abs_rel':>10}{'delta1':>9}")
    print("-" * 34)
    for k in names:
        print(f"{k:<14}{s[k]['abs_rel']:>10.5f}{s[k]['delta1']:>9.4f}")
    print(f"\ngain vs single: {summary['gain_pct']:+.2f}%")
    c = summary["residual_correlation"]
    print(f"\nresidual correlation, original vs flipped:")
    print(f"  overall     {c['all']:.3f}")
    print(f"  LOW freq    {c['low_freq']:.3f}   <- the global component")
    print(f"  HIGH freq   {c['high_freq']:.3f}")
    print("\n(multi-resolution gives ~0.67 overall; a lower LOW-freq value here means flips")
    print(" decorrelate the global error in a way that changing resolution does not.)")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
