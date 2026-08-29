#!/usr/bin/env python
"""Evaluate Lotus on Booster, broken down by material class.

Booster (Zama Ramirez et al., TPAMI 2023) is the benchmark this project needed and
NYUv2 could not provide:

  - 12 Mpx source images (4112x3008), so detail actually exists to be recovered -
    on NYUv2 at 640x480, processing_res=768 was already upsampling
  - dense ground truth ON specular and transparent surfaces, acquired by space-time
    stereo rather than an IR sensor. NYUv2's Kinect returns holes exactly there, and
    holes are excluded from scoring, so mirror errors were unmeasurable
  - manually annotated material masks, so error can be attributed to the surfaces
    that are actually hard instead of being averaged away

Phase 0 on NYUv2 found error density near 1.0 everywhere - no concentration to
attack. Booster is built so that difficulty concentrates on labelled regions, which
is what the per-class breakdown here measures.

Ground truth is disparity in pixels. Lotus emits affine-invariant disparity, so the
prediction is scale/shift-fitted to GT disparity (the same alignment the NYUv2
protocol uses) and both are inverted to relative depth. abs_rel and delta1 are
invariant to the remaining global scale, so no camera baseline is needed.

Uses the train split, whose GT is public; the mono test split withholds GT for
leaderboard submission.
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

from evaluation.util.alignment import align_depth_least_square
from evaluation.util.metric import abs_relative_difference, delta1_acc
from pipeline import LotusDPipeline
from utils.seed_all import seed_all

Image.MAX_IMAGE_PIXELS = None  # 12 Mpx frames trip PIL's decompression-bomb guard

# Material classes are documented only as "0, 1, 2 and 3" in the release README.
# Inspecting the masks: 0 covers walls/tables, 1 tiles and cabinet fronts, 2 the TV
# screen, 3 transparent bottles and mirrors. Difficulty is what the eval measures,
# so these labels are descriptive only.
CLASS_NAMES = {0: "class0 (lambertian)", 1: "class1", 2: "class2", 3: "class3"}


def parse_args():
    p = argparse.ArgumentParser(description="Lotus on Booster, per material class.")
    p.add_argument(
        "--data_root",
        type=str,
        default="D:/lotus/data/booster/extracted/train/balanced",
        help="Booster train/balanced root (scene folders with disp_00.npy + mask_cat.png).",
    )
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--output_dir", type=str, default="output/eval_booster_mono")
    p.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Lotus inference resolution (768 = official default; see docs/phase0_findings.md).",
    )
    p.add_argument(
        "--illum",
        type=str,
        default="im0",
        help="Illumination frame per scene, or 'all' to evaluate every frame.",
    )
    p.add_argument(
        "--eval_scale",
        type=int,
        default=2,
        help=(
            "Downsample factor for metric computation. GT is 12 Mpx; scoring at 1/2 "
            "keeps memory sane and does not change relative comparisons. 1 = full res."
        ),
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_samples(root: Path, illum: str):
    """Yield (scene, frame_name, rgb_path) for scenes that carry GT."""
    out = []
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (scene / "disp_00.npy").is_file() or not (scene / "mask_cat.png").is_file():
            continue
        cam = scene / "camera_00"
        if not cam.is_dir():
            continue
        frames = sorted(cam.glob("im*.png"))
        if illum != "all":
            frames = [f for f in frames if f.stem == illum] or frames[:1]
        out.extend((scene, f.stem, f) for f in frames)
    return out


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


def downsample(a: np.ndarray, k: int, nearest: bool = False) -> np.ndarray:
    if k <= 1:
        return a
    if nearest:
        return a[::k, ::k]
    h, w = a.shape[:2]
    im = Image.fromarray(a.astype(np.float32))
    return np.array(im.resize((w // k, h // k), Image.BILINEAR), dtype=np.float32)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    seed_all(args.seed)

    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"

    samples = list_samples(root, args.illum)
    if args.max_scenes > 0:
        keep = {s.name for s in sorted({s for s, _, _ in samples})}
        keep = set(sorted(keep)[: args.max_scenes])
        samples = [t for t in samples if t[0].name in keep]
    logging.info("samples: %d", len(samples))

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

    # summed |relative error| and pixel counts, overall and per material class
    acc = {"all": {"err": 0.0, "d1": 0.0, "n": 0}}
    for c in CLASS_NAMES:
        acc[c] = {"err": 0.0, "d1": 0.0, "n": 0}
    rows = []

    for scene, frame, rgb_path in tqdm(samples, desc="booster"):
        cpath = cache / scene.name / f"{frame}_pred.npy"
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        if cpath.is_file():
            pred = np.load(cpath).astype(np.float32)
        else:
            pred = predict_disparity(get_pipe(), rgb, args.timestep, args.processing_res, generator)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(cpath, pred.astype(np.float16))

        gt = np.load(scene / "disp_00.npy").astype(np.float32)
        occ = np.array(Image.open(scene / "mask_00.png"))
        cat = np.array(Image.open(scene / "mask_cat.png"))

        k = args.eval_scale
        pred = downsample(pred, k)
        gt = downsample(gt, k)
        occ = downsample(occ, k, nearest=True)
        cat = downsample(cat, k, nearest=True)

        valid = np.isfinite(gt) & (gt > 0) & (occ > 127) & np.isfinite(pred) & (pred > 0)
        if valid.sum() < 1000:
            continue

        # Lotus disparity is affine-invariant: fit it onto GT disparity, then invert
        # both to relative depth. abs_rel and delta1 are scale-invariant, so the
        # unknown focal*baseline factor never has to be recovered.
        aligned = align_depth_least_square(
            gt_arr=gt.astype(np.float64),
            pred_arr=pred.astype(np.float64),
            valid_mask_arr=valid,
            return_scale_shift=False,
        )
        aligned = np.clip(aligned, 1e-3, None)
        pred_depth = 1.0 / aligned
        gt_depth = 1.0 / np.clip(gt.astype(np.float64), 1e-3, None)

        pt = torch.from_numpy(pred_depth)
        gt_t = torch.from_numpy(gt_depth)
        row = {"scene": scene.name, "frame": frame}
        for key, m in [("all", valid)] + [(c, valid & (cat == c)) for c in CLASS_NAMES]:
            n = int(m.sum())
            if n < 500:
                continue
            mt = torch.from_numpy(m)
            a = float(abs_relative_difference(pt, gt_t, mt))
            d = float(delta1_acc(pt, gt_t, mt))
            acc[key]["err"] += a * n
            acc[key]["d1"] += d * n
            acc[key]["n"] += n
            row[f"abs_rel_{key}"] = a
            row[f"delta1_{key}"] = d
            row[f"px_{key}"] = n
        rows.append(row)

    if not rows:
        raise RuntimeError("No samples scored.")

    total_px = acc["all"]["n"]
    summary = {
        "num_samples": len(rows),
        "processing_res": args.processing_res,
        "eval_scale": args.eval_scale,
        "illum": args.illum,
        "core_model": args.core_model,
        "seed": args.seed,
        "half_precision": bool(args.half_precision),
        "classes": {},
    }
    for key in ["all"] + list(CLASS_NAMES):
        a = acc[key]
        if a["n"] == 0:
            continue
        summary["classes"][str(key)] = {
            "name": "all" if key == "all" else CLASS_NAMES[key],
            "abs_rel": a["err"] / a["n"],
            "delta1": a["d1"] / a["n"],
            "pixel_share": a["n"] / max(total_px, 1),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = sorted({k for r in rows for k in r})
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    base = summary["classes"]["all"]["abs_rel"]
    print(f"\nBooster train/balanced  processing_res={args.processing_res}  n={len(rows)}")
    print(f"{'region':<24}{'px share':>10}{'abs_rel':>10}{'delta1':>9}{'vs all':>9}")
    print("-" * 62)
    for key in ["all"] + list(CLASS_NAMES):
        c = summary["classes"].get(str(key))
        if not c:
            continue
        rel = c["abs_rel"] / base if base else 1.0
        print(
            f"{c['name']:<24}{c['pixel_share']*100:>9.1f}%{c['abs_rel']:>10.5f}"
            f"{c['delta1']:>9.4f}{rel:>8.2f}x"
        )
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
