#!/usr/bin/env python
"""Tiled high-resolution Lotus inference on NYUv2 (training-free).

Phase 1-2 of docs/lotus_improvement_plan.md, designed in docs/tiled_inference_plan.md.

Runs Lotus once on the whole image and once per overlapping tile, affine-fits each
tile onto the global pass, and blends. No training, so the official model can only be
added to, never degraded - which matters because every fine-tuning attempt in this
project cost more than its conditioning gained.

Compare against the corrected baseline: processing_res=768 gives abs_rel 0.05000
(output/eval_baseline_official_654_res768). Comparing against the 512 number (0.05543)
would credit this method with the 9.8% that simply came from fixing a misconfiguration.

Tile predictions are cached per (grid, overlap, tile_res), so fusion-parameter
ablations (--fusion / --sigma / --ramp_px / --trim) re-run without any inference.
"""

from __future__ import annotations

import argparse
import csv
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

from eval_regressor_predepth_nyuv2 import list_nyu_pairs, score_prediction
from pipeline import LotusDPipeline
from utils.seed_all import seed_all
from utils.tiled_depth_fusion import (
    align_tile_to_global,
    feather_weight,
    fuse_average,
    fuse_detail_transfer,
    latent_density_multiplier,
    make_tiles,
)


def parse_args():
    p = argparse.ArgumentParser(description="Tiled Lotus inference on NYUv2.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--output_dir", type=str, default="output/eval_tiled_depth")

    # geometry
    p.add_argument("--grid", type=int, default=2, help="grid x grid tiles.")
    p.add_argument("--overlap", type=float, default=0.25, help="Fractional tile overlap.")
    p.add_argument(
        "--tile_res",
        type=int,
        default=768,
        help=(
            "processing_res for tile passes. Keep at the global value to buy latent "
            "density; set it lower to match the global density (control C2)."
        ),
    )
    p.add_argument("--global_res", type=int, default=768, help="processing_res for the global pass.")

    # fusion
    p.add_argument("--fusion", choices=["f2", "f1", "none"], default="f2",
                   help="f2=detail transfer (default), f1=weighted average, none=global only.")
    p.add_argument("--sigma", type=float, default=8.0, help="F2 low-pass sigma in source pixels.")
    p.add_argument("--ramp_px", type=int, default=32, help="Feather ramp width at tile seams.")
    p.add_argument("--trim", type=float, default=0.25, help="Outlier fraction trimmed from the affine fit.")

    p.add_argument("--global_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--tile_cache_dir", type=str, default="D:/lotus/data/tiled_cache")
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def predict_disparity(pipe, rgb_np, timestep, processing_res, generator):
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = (image / 127.5 - 1.0).to(device)
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device_type=device.type)
    with ctx:
        out = pipe(
            rgb_in=image,
            prompt="",
            num_inference_steps=1,
            generator=generator,
            output_type="np",
            timesteps=[timestep],
            task_emb=task_emb,
            processing_res=processing_res,
            match_input_res=True,
        ).images[0]
    return (out.mean(axis=-1) if out.ndim == 3 else out).astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    seed_all(args.seed)

    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_cache = Path(args.global_cache_dir) / f"res{args.global_res}"
    tile_cache = Path(args.tile_cache_dir) / f"g{args.grid}_o{args.overlap}_r{args.tile_res}"

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

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

    rows = []
    density = None

    for rgb_path, depth_path in tqdm(pairs, desc="tiled_depth"):
        rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = rgb_np.shape[:2]
        rel = rgb_path.relative_to(rgb_dir)

        # --- global pass (shares the cache built by eval_object_oracle_ceiling.py) ---
        gpath = global_cache / rel.parent / f"{rel.stem}_pred.npy"
        if gpath.is_file():
            g_disp = np.load(gpath).astype(np.float32)
        else:
            g_disp = predict_disparity(get_pipe(), rgb_np, args.timestep, args.global_res, generator)
            gpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(gpath, g_disp.astype(np.float16))
        if g_disp.shape != (h, w):
            g_disp = np.array(
                Image.fromarray(g_disp).resize((w, h), Image.BILINEAR), dtype=np.float32
            )

        if args.fusion == "none":
            fused = g_disp
        else:
            tiles = make_tiles(h, w, args.grid, args.overlap)
            if density is None:
                density = latent_density_multiplier(h, w, tiles) * (args.tile_res / args.global_res)

            tpath = tile_cache / rel.parent / f"{rel.stem}_tiles.npz"
            if tpath.is_file():
                data = np.load(tpath)
                crops = [data[f"t{i}"].astype(np.float32) for i in range(len(tiles))]
            else:
                crops = [
                    predict_disparity(
                        get_pipe(), rgb_np[t.slice], args.timestep, args.tile_res, generator
                    )
                    for t in tiles
                ]
                tpath.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    tpath, **{f"t{i}": c.astype(np.float16) for i, c in enumerate(crops)}
                )

            aligned = [
                align_tile_to_global(t, c, g_disp, ramp_px=args.ramp_px, trim=args.trim)
                for t, c in zip(tiles, crops)
            ]
            weights = [feather_weight(t, h, w, args.ramp_px) for t in tiles]
            fused = (
                fuse_detail_transfer(g_disp, tiles, aligned, weights, sigma=args.sigma)
                if args.fusion == "f2"
                else fuse_average(g_disp, tiles, aligned, weights)
            )

        absrel, d1 = score_prediction(fused.astype(np.float64), gt, use_eigen_crop=True)
        if absrel is None:
            continue
        base_absrel, base_d1 = score_prediction(g_disp.astype(np.float64), gt, use_eigen_crop=True)
        rows.append(
            {
                "filename": str(rel).replace("\\", "/"),
                "abs_rel": absrel,
                "delta1": d1,
                "abs_rel_global_only": base_absrel,
                "delta1_global_only": base_d1,
            }
        )

    if not rows:
        raise RuntimeError("No images scored.")

    mean = lambda k: float(np.mean([r[k] for r in rows]))
    summary = {
        "num_images": len(rows),
        "grid": args.grid,
        "overlap": args.overlap,
        "tile_res": args.tile_res,
        "global_res": args.global_res,
        "fusion": args.fusion,
        "sigma": args.sigma,
        "ramp_px": args.ramp_px,
        "trim": args.trim,
        "latent_density_multiplier": density,
        "seed": args.seed,
        "half_precision": bool(args.half_precision),
        "abs_rel": mean("abs_rel"),
        "delta1": mean("delta1"),
        "abs_rel_global_only": mean("abs_rel_global_only"),
        "delta1_global_only": mean("delta1_global_only"),
    }
    summary["gain_vs_global"] = summary["abs_rel_global_only"] - summary["abs_rel"]
    summary["gain_pct"] = (
        100.0 * summary["gain_vs_global"] / summary["abs_rel_global_only"]
        if summary["abs_rel_global_only"]
        else 0.0
    )
    win = sum(1 for r in rows if r["abs_rel"] < r["abs_rel_global_only"])
    summary["win_rate_vs_global"] = win / len(rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\ngrid {args.grid}x{args.grid}  overlap {args.overlap}  tile_res {args.tile_res}  "
        f"fusion {args.fusion}"
        + (f"  sigma {args.sigma}" if args.fusion == "f2" else "")
    )
    if density:
        print(f"latent density vs global: {density:.2f}x")
    print(f"{'':<18}{'abs_rel':>10}{'delta1':>9}")
    print(f"{'global only':<18}{summary['abs_rel_global_only']:>10.5f}{summary['delta1_global_only']:>9.4f}")
    print(f"{'tiled':<18}{summary['abs_rel']:>10.5f}{summary['delta1']:>9.4f}")
    print(
        f"gain {summary['gain_vs_global']:+.5f} ({summary['gain_pct']:+.1f}%)  "
        f"win rate {summary['win_rate_vs_global']*100:.1f}%  n={len(rows)}"
    )
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
