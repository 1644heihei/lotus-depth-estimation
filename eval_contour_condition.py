#!/usr/bin/env python
"""Score a contour-conditioned Lotus against the criteria fixed before training.

The contour is fed as extra UNet input channels, so evaluation has to feed it too - the
pipeline pads unknown channel counts with zeros, and a model trained on contours scored
with a zero channel would look like the conditioning did nothing.

Three inputs per model, because a gain has two innocent explanations that have to be ruled
out separately:

  none       an empty channel. What the model does with nothing, which is also the
             condition the untrained Lotus baseline was measured under.
  contour    this image's SAM contour. The treatment.
  shuffled   another image's contour. The same amount of contour describing the wrong
             scene - if this scores like `contour`, the channel is being used as a generic
             perturbation rather than as position.

Scored on BF1 above all, since that is what a contour is supposed to fix, alongside abs_rel
so a boundary gain bought with accuracy is visible as such. Untrained references on NYUv2
654: BF1 0.0693, abs_rel 0.05000, off-edge recall 23.9%.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_boundary_f1 import boundary_f1
from eval_edge_dependence import within
from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path, align_to_gt, load_or_build_masks, score
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from pipeline import LotusDPipeline
from utils.contour_condition import contour_of
from utils.expanded_conv_in import extra_channel_energy, load_conv_in
from utils.lora_eval_loader import add_lora_args, apply_lora, resolve_lora_dir, run_tag_for

MODES = ["none", "contour", "shuffled"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate contour-conditioned Lotus.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/sam_seg")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/contour_pred_cache")
    p.add_argument("--output_dir", type=str, default="output/eval_contour_condition")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--contour_width", type=int, default=1)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--canny", type=int, nargs=2, default=(50, 150))
    p.add_argument("--modes", type=str, nargs="+", default=MODES, choices=MODES)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    add_lora_args(p)
    return p.parse_args()


@torch.no_grad()
def predict(pipe, rgb_np, contour, timestep, processing_res, generator):
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
            contour=contour,
        ).images[0]
    return (out.mean(axis=-1) if out.ndim == 3 else out).astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    if args.lora_path:
        # conv_in first: the pipeline is built from the base model, whose conv_in is 4
        # channels wide, so the contour would be dropped before the adapter is even applied
        load_conv_in(pipe.unet, resolve_lora_dir(args.lora_path))
        pipe.unet.config.in_channels = pipe.unet.conv_in.in_channels
        pipe.unet.to(dtype=dtype, device=device)
    apply_lora(pipe, args.lora_path)
    tag = run_tag_for(args)
    energy = extra_channel_energy(pipe.unet, 4)
    logging.info("Weights: %s   cache tag: %s   conv_in in_channels=%d   "
                 "extra-channel |w|max=%.4e",
                 args.lora_path or args.core_model, tag,
                 pipe.unet.conv_in.in_channels, energy)
    if args.lora_path and energy == 0.0:
        raise RuntimeError(
            "The contour channel's weights are zero, so the contour cannot affect the "
            "output. Scoring would compare three identical models."
        )

    # every image's contour up front: `shuffled` needs a neighbour's, and building them
    # twice would let the two runs disagree about what a neighbour is
    contours = []
    for rgb_path, depth_path in pairs:
        h, w = np.array(Image.open(depth_path)).shape[:2]
        seg = list(load_or_build_masks(rgb_path, rgb_dir, Path(args.mask_cache_dir), None,
                                       np.empty((h, w, 3), np.uint8), args.detection_score_thr))
        c = np.full((h, w), -1.0, np.float32)
        if seg:
            c[contour_of(np.any(np.stack(seg), axis=0), args.contour_width)] = 1.0
        contours.append(c)
    frac = float(np.mean([(c > 0).mean() for c in contours]))
    logging.info("contour maps built: %d   mean on-contour fraction %.4f", len(contours), frac)
    if frac <= 0:
        raise RuntimeError(f"Every contour is empty; check --mask_cache_dir {args.mask_cache_dir}")

    thresholds = np.linspace(5.0, 25.0, 11)
    weights = thresholds / thresholds.sum()
    modes = [m for m in MODES if m in set(args.modes)]
    acc = {m: {"absrel": [], "d1": [], "bf1": [], "on": [0, 0], "off": [0, 0]} for m in modes}

    for idx, (rgb_path, depth_path) in enumerate(tqdm(pairs, desc="contour_eval")):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        if valid.sum() < 100:
            continue
        rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
        grey = np.array(Image.open(rgb_path).convert("L"))
        edge = within(cv2.Canny(grey, *args.canny).astype(bool) & valid, 1)
        gt_d = discontinuities(gt, valid, args.t)
        gt_on, gt_off = gt_d & edge, gt_d & ~edge

        maps = {
            "none": np.full((h, w), -1.0, np.float32),
            "contour": contours[idx],
            "shuffled": contours[(idx + 1) % len(contours)],
        }
        for m in modes:
            cache = Path(args.pred_cache_dir) / f"res{args.processing_res}" / tag / m
            cp = _cache_path(rgb_path, rgb_dir, cache, "_pred.npy")
            if cp.is_file():
                pred = np.load(cp).astype(np.float64)
            else:
                cm = maps[m]
                if cm.shape != (h, w):
                    cm = cv2.resize(cm, (w, h), interpolation=cv2.INTER_NEAREST)
                g = torch.Generator(device=device).manual_seed(args.seed)
                pred = predict(
                    pipe, rgb_np,
                    torch.from_numpy(cm)[None, None].to(device=device, dtype=dtype),
                    args.timestep, args.processing_res, g,
                )
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.save(cp, pred.astype(np.float16))
                pred = pred.astype(np.float16).astype(np.float64)
            base = align_to_gt(pred, gt, valid)
            if base is None:
                continue
            a, d1 = score(base, gt, valid)
            acc[m]["absrel"].append(a)
            acc[m]["d1"].append(d1)
            c = boundary_f1(base, gt, valid, thresholds)
            ok = np.isfinite(c)
            acc[m]["bf1"].append(float((c[ok] * weights[ok]).sum() / weights[ok].sum())
                                 if ok.any() else np.nan)
            lo_d = within(discontinuities(base, valid, args.t), 1)
            acc[m]["on"][0] += int((gt_on & lo_d).sum())
            acc[m]["on"][1] += int(gt_on.sum())
            acc[m]["off"][0] += int((gt_off & lo_d).sum())
            acc[m]["off"][1] += int(gt_off.sum())

    summary = {"n_images": len(acc[modes[0]]["absrel"]), "lora_path": args.lora_path,
               "run_tag": tag, "mean_contour_frac": frac, "modes": {}}
    for m in modes:
        e = acc[m]
        summary["modes"][m] = {
            "abs_rel": float(np.mean(e["absrel"])), "delta1": float(np.mean(e["d1"])),
            "bf1": float(np.nanmean(e["bf1"])),
            "recall_on_edge": e["on"][0] / max(e["on"][1], 1),
            "recall_off_edge": e["off"][0] / max(e["off"][1], 1),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    M = summary["modes"]
    print(f"\nContour conditioning  {tag}  n={summary['n_images']}")
    print(f"\n{'input':<12}{'BF1':>9}{'abs_rel':>10}{'delta1':>9}"
          f"{'recall ON':>11}{'recall OFF':>12}")
    print("-" * 63)
    print(f"{'(untrained)':<12}{0.0693:>9.4f}{0.05000:>10.5f}{0.9711:>9.4f}"
          f"{47.7:>10.1f}%{23.9:>11.1f}%")
    for m in modes:
        d = M[m]
        print(f"{m:<12}{d['bf1']:>9.4f}{d['abs_rel']:>10.5f}{d['delta1']:>9.4f}"
              f"{d['recall_on_edge']*100:>10.1f}%{d['recall_off_edge']*100:>11.1f}%")
    if "contour" in M and "shuffled" in M:
        print(f"\ncontour vs shuffled: BF1 {M['contour']['bf1'] - M['shuffled']['bf1']:+.4f}"
              f"   <- the contour describing THIS image rather than any image")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
