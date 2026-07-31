#!/usr/bin/env python
"""Conditioned NYUv2 evaluation for Approach-A detail models.

Uses offline artifacts (YOLO / pre-depth / valid) under --detail_artifacts_dir,
rebuilds class/size maps at --detection_score_thr, then scores absrel / delta1
with least-squares disparity alignment (same protocol as Lotus NYUv2 eval).
"""

from __future__ import annotations

import argparse
import csv
import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from evaluation.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from evaluation.util.metric import abs_relative_difference, delta1_acc
from pipeline import LotusDPipeline
from utils.object_condition import class_map_to_tensor
from utils.object_detection_cache import detections_to_mask, load_detections
from utils.object_pre_depth import load_pre_depth_artifacts
from utils.object_size_condition import rasterize_class_and_size_maps, size_map_to_tensor
from utils.seed_all import seed_all


def parse_args():
    p = argparse.ArgumentParser(description="Conditioned NYUv2 eval for detail Lotus-D.")
    p.add_argument("--detail_model", type=str, required=True)
    p.add_argument(
        "--detail_artifacts_dir",
        type=str,
        default="D:/lotus/data/nyuv2_detail_artifacts/test",
        help="Offline NYUv2 detail artifact root (same as build output_dir).",
    )
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument(
        "--unconditioned",
        action="store_true",
        help="Zero-out all object/pre-depth conditions (condition-free baseline).",
    )
    p.add_argument("--save_pred", action="store_true")
    return p.parse_args()


def list_nyu_pairs(rgb_dir: Path):
    pairs = []
    for rgb_path in sorted(rgb_dir.rglob("rgb_*.png")):
        depth_path = rgb_path.parent / rgb_path.name.replace("rgb_", "depth_", 1)
        if depth_path.is_file():
            pairs.append((rgb_path, depth_path))
    return pairs


def eigen_valid_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[45:471, 41:601] = True
    return m


@torch.no_grad()
def predict_detail(
    pipe,
    rgb_np: np.ndarray,
    *,
    pre_depth_norm: np.ndarray | None,
    valid_mask: np.ndarray | None,
    class_map: np.ndarray | None,
    size_w: np.ndarray | None,
    size_h: np.ndarray | None,
    timestep: int,
    processing_res: int | None,
    generator: torch.Generator,
):
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = image / 127.5 - 1.0
    image = image.to(device)

    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    kwargs = dict(
        rgb_in=image,
        prompt="",
        num_inference_steps=1,
        generator=generator,
        output_type="np",
        timesteps=[timestep],
        task_emb=task_emb,
        processing_res=processing_res,
        match_input_res=True,
    )

    extra = int(pipe.unet.config.in_channels) - 4
    if extra > 0 and pre_depth_norm is not None and valid_mask is not None:
        pre_t = torch.from_numpy(pre_depth_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        pre_t = pre_t.repeat(1, 3, 1, 1).to(device=device, dtype=image.dtype)
        valid_t = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(
            device=device, dtype=image.dtype
        )
        kwargs["pre_depth"] = pre_t
        kwargs["pre_depth_valid_mask"] = valid_t

        if extra in (8, 9) and class_map is not None:
            class_t = torch.from_numpy(class_map_to_tensor(class_map)).unsqueeze(0).unsqueeze(0)
            kwargs["class_map"] = class_t.to(device=device, dtype=image.dtype)
        if extra == 8 and size_w is not None and size_h is not None:
            kwargs["size_w"] = torch.from_numpy(size_map_to_tensor(size_w)).unsqueeze(0).unsqueeze(0).to(
                device=device, dtype=image.dtype
            )
            kwargs["size_h"] = torch.from_numpy(size_map_to_tensor(size_h)).unsqueeze(0).unsqueeze(0).to(
                device=device, dtype=image.dtype
            )

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type=device.type)

    with autocast_ctx:
        pred = pipe(**kwargs).images[0]
    return pred.mean(axis=-1).astype(np.float32)


def score_prediction(pred_disp: np.ndarray, gt_depth: np.ndarray):
    pred = pred_disp.astype(np.float64)
    gt = gt_depth.astype(np.float64)
    if pred.shape != gt.shape:
        pred = np.array(
            Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR),
            dtype=np.float64,
        )
    valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0)
    valid &= eigen_valid_mask(*gt.shape[:2])
    gt_disp, gt_nn = depth2disparity(depth=gt, return_mask=True)
    valid_nn = valid & gt_nn & (pred > 0)
    if valid_nn.sum() < 100:
        return None, None
    disp_aligned, _, _ = align_depth_least_square(
        gt_arr=gt_disp,
        pred_arr=pred,
        valid_mask_arr=valid_nn,
        return_scale_shift=True,
        max_resolution=None,
    )
    depth_pred = np.clip(disparity2depth(np.clip(disp_aligned, 1e-3, None)), 1e-3, 10.0)
    pred_t = torch.from_numpy(depth_pred)
    gt_t = torch.from_numpy(gt)
    valid_t = torch.from_numpy(valid)
    return float(abs_relative_difference(pred_t, gt_t, valid_t)), float(delta1_acc(pred_t, gt_t, valid_t))


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_artifacts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pred:
        (out_dir / "pred").mkdir(parents=True, exist_ok=True)

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images and args.max_images > 0:
        pairs = pairs[: args.max_images]
    if not pairs:
        raise FileNotFoundError(f"No rgb/depth pairs under {rgb_dir}")

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    logging.info("Loading detail model: %s", args.detail_model)
    pipe = LotusDPipeline.from_pretrained(args.detail_model, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    in_ch = int(pipe.unet.config.in_channels)
    logging.info("UNet in_channels=%d  unconditioned=%s  score_thr=%.2f", in_ch, args.unconditioned, args.detection_score_thr)

    rows = []
    missing_artifacts = 0
    n_zero_det = 0

    for rgb_path, depth_path in tqdm(pairs, desc="nyuv2_conditioned"):
        rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = rgb_np.shape[:2]

        pre_depth = valid = class_map = size_w = size_h = None
        n_det = 0

        if not args.unconditioned and in_ch > 4:
            pre_depth, valid = load_pre_depth_artifacts(rgb_path, detail_root)
            if pre_depth is None or valid is None:
                missing_artifacts += 1
                logging.warning("Missing artifacts for %s", rgb_path)
                # Skip rather than silently evaluating unconditioned.
                continue

            dets = [d for d in load_detections(rgb_path, detail_root) if d.score >= args.detection_score_thr]
            n_det = len(dets)
            if n_det == 0:
                n_zero_det += 1
            keep = detections_to_mask(dets, h, w)
            valid = (valid.astype(np.float32) * keep).astype(np.float32)
            class_map, size_w, size_h = rasterize_class_and_size_maps(dets, h, w)

            # Resize condition maps if needed (should already match).
            if pre_depth.shape[:2] != (h, w):
                pre_depth = np.array(Image.fromarray(pre_depth).resize((w, h), Image.BILINEAR))
            if valid.shape[:2] != (h, w):
                valid = np.array(Image.fromarray(valid).resize((w, h), Image.NEAREST))

        pred = predict_detail(
            pipe,
            rgb_np,
            pre_depth_norm=pre_depth,
            valid_mask=valid,
            class_map=class_map,
            size_w=size_w,
            size_h=size_h,
            timestep=args.timestep,
            processing_res=args.processing_res,
            generator=generator,
        )
        absrel, d1 = score_prediction(pred, gt)
        if absrel is None:
            logging.warning("Skip metrics (too few valid pixels): %s", rgb_path.name)
            continue

        rel = str(rgb_path.relative_to(rgb_dir)).replace("\\", "/")
        rows.append(
            {
                "filename": rel,
                "abs_relative_difference": absrel,
                "delta1_acc": d1,
                "n_detections": n_det,
            }
        )
        if args.save_pred:
            np.save(out_dir / "pred" / f"{rgb_path.stem}.npy", pred)

    if not rows:
        raise RuntimeError(
            "No evaluated samples. If artifacts are missing, run "
            "train_scripts/build_nyuv2_detail_artifacts.ps1 first."
        )

    mean_abs = float(np.mean([r["abs_relative_difference"] for r in rows]))
    mean_d1 = float(np.mean([r["delta1_acc"] for r in rows]))

    csv_path = out_dir / "per_sample_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = (
        f"samples={len(rows)}\n"
        f"abs_relative_difference={mean_abs:.6f}\n"
        f"delta1_acc={mean_d1:.6f}\n"
        f"missing_artifacts_skipped={missing_artifacts}\n"
        f"zero_detection_images={n_zero_det}\n"
        f"unconditioned={args.unconditioned}\n"
        f"detection_score_thr={args.detection_score_thr}\n"
        f"detail_model={args.detail_model}\n"
        f"in_channels={in_ch}\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    logging.info("Wrote %s and summary.txt", csv_path)


if __name__ == "__main__":
    main()
