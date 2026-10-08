#!/usr/bin/env python
"""Wall-clock cost of each stage, so the gain can be read against what it costs.

The selector is 1.93M parameters and the sharpening is a dilation and two filters, so the
part this work adds is almost free. What is not free is SAM: the ablation measured it at
5.19 points of the 12.49 (42% of the gain), and it is a 641M model run at its own
resolution on every frame. A reader deciding whether to adopt this needs both numbers
together, so they are measured here on the same machine in one run.

Measured per stage, after a warm-up pass, on real frames rather than synthetic input -
SAM's automatic mode runs a point grid and its cost depends on how many regions the image
actually has.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contour_selector import (UNet, build_input, cache_path, in_channels,
                              load_candidates, load_prediction, top_share_mask)
from eval_yolo_contour_sharpening import sharpen
from utils.align_space import add_align_space_arg, get_aligner
from utils.eval_frames import (add_dataset_args, depth_scale, list_frames,
                               valid_mask)

NYU = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar"


def parse_args():
    p = argparse.ArgumentParser(description="Per-stage wall clock for the pipeline.")
    p.add_argument("--rgb_dir", type=str, default=f"{NYU}/test")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/marigold_v2_pred/res640")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/sam_auto48")
    p.add_argument("--checkpoint", type=str, default="output/contour_selector/best.pt")
    p.add_argument("--out", type=str, default="output/bench_pipeline.json")
    p.add_argument("--n", type=int, default=20, help="Frames timed per stage.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--band_px", type=int, default=2)
    p.add_argument("--fill_radius", type=int, default=3)
    p.add_argument("--retention", type=float, default=0.06)
    p.add_argument("--points_per_crop", type=int, default=48)
    p.add_argument("--points_per_batch", type=int, default=64)
    p.add_argument("--sam_model", type=str, default="facebook/sam-vit-huge")
    p.add_argument("--skip_sam", action="store_true",
                   help="Time only the stages this work adds, when SAM's weights are "
                        "not available on the machine doing the timing.")
    add_align_space_arg(p)
    add_dataset_args(p)
    return p.parse_args()


def timed(fn, frames, warmup, desc):
    """Seconds per frame, after a warm-up that is not counted."""
    for f in frames[:warmup]:
        fn(f)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    per = []
    for f in tqdm(frames, desc=desc, leave=False):
        t0 = time.perf_counter()
        fn(f)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per.append(time.perf_counter() - t0)
    a = np.array(per)
    return {"mean_s": float(a.mean()), "median_s": float(np.median(a)),
            "p90_s": float(np.percentile(a, 90)), "n": len(a)}


def main():
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    pred_dir, mask_dir = Path(args.pred_cache_dir), Path(args.mask_cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    align = get_aligner(args.align_space)
    ker = np.ones((2 * args.band_px + 1, 2 * args.band_px + 1), np.uint8)

    pairs = list_frames(args)[: args.n + args.warmup]
    assert len(pairs) > args.warmup, "not enough frames"

    # Everything each stage needs, loaded once: disk reads are not what is being timed.
    ctx = []
    for rgb_path, depth_path in pairs:
        gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(args.dataset)
        valid = valid_mask(args.dataset, gt)
        base = align(np.load(cache_path(rgb_path, rgb_dir, pred_dir, "_pred.npy")
                             ).astype(np.float64), gt, valid)
        cand = load_candidates(rgb_path, rgb_dir, mask_dir, valid)
        pred = load_prediction(rgb_path, rgb_dir, pred_dir, "log")
        ctx.append({"rgb_path": rgb_path, "gt": gt, "valid": valid, "base": base,
                    "cand": cand, "pred": pred,
                    "rgb": np.asarray(Image.open(rgb_path).convert("RGB")),
                    "pil": Image.open(rgb_path).convert("RGB")})
    ctx = [c for c in ctx if c["base"] is not None and c["cand"].any()]

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = UNet(in_channels(), ck["args"]["base_ch"]).to(device).eval()
    model.load_state_dict(ck["model"])

    results = {}

    @torch.no_grad()
    def run_selector(c):
        x = build_input(c["rgb_path"], c["pred"], c["cand"])[None]
        score = model(torch.from_numpy(x).to(device))[0, 0].float().cpu().numpy()
        return top_share_mask(score, c["cand"], args.retention)

    # Attach each frame's selection to the frame, so the sharpening timer is not also
    # timing a lookup (and so it cannot silently pair the wrong mask with a frame).
    for c in ctx:
        c["keep"] = run_selector(c)
    results["selector_unet"] = timed(run_selector, ctx, args.warmup, "selector")
    results["sharpening"] = timed(
        lambda c: sharpen(c["base"], c["keep"], c["valid"], ker,
                          args.band_px, args.fill_radius),
        ctx, args.warmup, "sharpen")

    if not args.skip_sam:
        from transformers import pipeline
        gen = pipeline("mask-generation", model=args.sam_model,
                       device=0 if device.type == "cuda" else -1)
        results["sam_vit_h"] = timed(
            lambda c: gen(c["pil"], points_per_batch=args.points_per_batch,
                          points_per_crop=args.points_per_crop),
            ctx, args.warmup, "sam")

    meta = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "python": platform.python_version(),
            "frames": len(ctx), "points_per_crop": args.points_per_crop,
            "selector_params": sum(p.numel() for p in model.parameters())}
    out = {"meta": meta, "stages": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{meta['device']}   {len(ctx)} frames")
    print(f"{'stage':18s} {'mean':>9} {'median':>9} {'p90':>9}")
    for k, v in results.items():
        print(f"{k:18s} {v['mean_s']*1000:8.1f}ms {v['median_s']*1000:8.1f}ms "
              f"{v['p90_s']*1000:8.1f}ms")
    added = results["selector_unet"]["mean_s"] + results["sharpening"]["mean_s"]
    print(f"\nthis work adds {added*1000:.1f}ms/frame")
    if "sam_vit_h" in results:
        print(f"SAM            {results['sam_vit_h']['mean_s']*1000:.1f}ms/frame  "
              f"({results['sam_vit_h']['mean_s']/max(added,1e-9):.0f}x the rest)")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
