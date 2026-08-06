#!/usr/bin/env python
"""NYUv2 evaluation with regressor-based pre-depth + detail Lotus-D."""

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

from evaluation.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from evaluation.util.metric import abs_relative_difference, delta1_acc
from pipeline import LotusDPipeline
from utils.object_depth_regressor import ObjectDepthRegressorBundle
from utils.object_detection_cache import detections_to_mask, load_detections
from utils.object_pre_depth import CoreDepthPredictor, load_pre_depth_artifacts
from utils.object_pre_depth_regressor import build_pre_depth_for_rgb
from utils.seed_all import seed_all


def parse_args():
    p = argparse.ArgumentParser(description="NYUv2 eval with regressor pre-depth.")
    p.add_argument("--detail_model", type=str, required=True)
    p.add_argument("--regressor_dir", type=str, required=True)
    p.add_argument(
        "--regressor_model_type",
        choices=[
            "config",
            "mlp",
            "linear",
            "ensemble",
            "baseline_global",
            "baseline_per_class",
        ],
        default="config",
    )
    p.add_argument(
        "--detail_artifacts_dir",
        type=str,
        default="D:/lotus/data/nyuv2_detail_artifacts/test",
    )
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument(
        "--dataset", choices=["nyuv2", "scannet"], default="nyuv2"
    )
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--condition_mode",
        type=str,
        choices=["regressor", "cached", "unconditioned"],
        default="regressor",
    )
    p.add_argument("--pre_depth_root", type=str, default=None)
    p.add_argument("--detection_score_thr", type=float, default=None)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_nyu_pairs(rgb_dir: Path):
    pairs = []
    for rgb_path in sorted(rgb_dir.rglob("rgb_*.png")):
        depth_path = rgb_path.parent / rgb_path.name.replace("rgb_", "depth_", 1)
        if depth_path.is_file():
            pairs.append((rgb_path, depth_path))
    return pairs


def list_scannet_pairs(rgb_dir: Path):
    pairs = []
    for rgb_path in sorted(rgb_dir.glob("scene*/color/*.jpg")):
        depth_path = rgb_path.parent.parent / "depth" / f"{rgb_path.stem}.png"
        if depth_path.is_file():
            pairs.append((rgb_path, depth_path))
    return pairs


def eigen_valid_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[45:471, 41:601] = True
    return m


def score_prediction(
    pred_disp: np.ndarray, gt_depth: np.ndarray, *, use_eigen_crop: bool = True
):
    pred = pred_disp.astype(np.float64)
    gt = gt_depth.astype(np.float64)
    if pred.shape != gt.shape:
        pred = np.array(
            Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR),
            dtype=np.float64,
        )
    valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0)
    if use_eigen_crop:
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


@torch.no_grad()
def predict_detail(pipe, rgb_np, pre_depth_norm, valid_mask, timestep, processing_res, generator):
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
    if torch.backends.mps.is_available():
        ctx = nullcontext()
    else:
        ctx = torch.autocast(device_type=device.type)
    with ctx:
        pred = pipe(**kwargs).images[0]
    return pred.mean(axis=-1).astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_artifacts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = (
        list_nyu_pairs(rgb_dir)
        if args.dataset == "nyuv2"
        else list_scannet_pairs(rgb_dir)
    )
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    detail_pipe = LotusDPipeline.from_pretrained(args.detail_model, torch_dtype=dtype).to(device)
    detail_pipe.set_progress_bar_config(disable=True)
    in_ch = int(detail_pipe.unet.config.in_channels)
    if in_ch != 9:
        raise ValueError(
            f"Regressor pre-depth evaluation requires a 9ch detail model, got in_channels={in_ch}."
        )

    core_pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=dtype).to(device)
    core_pipe.set_progress_bar_config(disable=True)
    regressor = ObjectDepthRegressorBundle.load(args.regressor_dir, device=device)
    if args.regressor_model_type != "config":
        regressor.model_type = args.regressor_model_type
    detection_score_thr = (
        float(args.detection_score_thr)
        if args.detection_score_thr is not None
        else float(regressor.config.get("detection_score_thr", 0.5))
    )
    processing_res = (
        args.processing_res
        if args.processing_res is not None
        else regressor.config.get("processing_res")
    )
    core_predictor = CoreDepthPredictor(
        core_pipe, processing_res=processing_res, generator=generator
    )

    rows = []
    roi_absrels = []
    for rgb_path, depth_path in tqdm(pairs, desc="nyuv2_regressor_predepth"):
        rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = rgb_np.shape[:2]

        pre_depth = valid = None
        n_det = 0
        if args.condition_mode == "regressor":
            dets = [
                d
                for d in load_detections(rgb_path, detail_root)
                if d.score >= detection_score_thr
            ]
            n_det = len(dets)
            pre_depth, valid, _ = build_pre_depth_for_rgb(
                rgb_np, dets, regressor, core_predictor
            )
        elif args.condition_mode == "cached":
            if not args.pre_depth_root:
                raise ValueError("--pre_depth_root is required for cached mode")
            pre_depth, valid = load_pre_depth_artifacts(
                rgb_path, args.pre_depth_root
            )

        pred = predict_detail(
            detail_pipe,
            rgb_np,
            pre_depth,
            valid,
            args.timestep,
            processing_res,
            generator,
        )
        absrel, d1 = score_prediction(
            pred, gt, use_eigen_crop=args.dataset == "nyuv2"
        )
        if absrel is None:
            continue

        rel = str(rgb_path.relative_to(rgb_dir)).replace("\\", "/")
        rows.append({"filename": rel, "abs_relative_difference": absrel, "delta1_acc": d1, "n_detections": n_det})

        if n_det > 0 and valid is not None:
            spatial_mask = (
                eigen_valid_mask(h, w)
                if args.dataset == "nyuv2"
                else np.ones((h, w), dtype=bool)
            )
            vm = spatial_mask & (valid > 0.5)
            gt_valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & vm
            if gt_valid.sum() > 100:
                gt_disp, gt_nn = depth2disparity(depth=gt, return_mask=True)
                valid_nn = gt_valid & gt_nn & (pred > 0)
                if valid_nn.sum() > 100:
                    disp_aligned, _, _ = align_depth_least_square(
                        gt_arr=gt_disp,
                        pred_arr=pred.astype(np.float64),
                        valid_mask_arr=valid_nn,
                        return_scale_shift=True,
                    )
                    depth_pred = np.clip(disparity2depth(np.clip(disp_aligned, 1e-3, None)), 1e-3, 10.0)
                    pred_t = torch.from_numpy(depth_pred)
                    gt_t = torch.from_numpy(gt)
                    vm_t = torch.from_numpy(gt_valid)
                    roi_absrels.append(float(abs_relative_difference(pred_t, gt_t, vm_t)))

    if not rows:
        raise RuntimeError("No scored images.")

    mean_absrel = float(np.mean([r["abs_relative_difference"] for r in rows]))
    mean_d1 = float(np.mean([r["delta1_acc"] for r in rows]))
    summary = {
        "detail_model": args.detail_model,
        "regressor_dir": args.regressor_dir,
        "regressor_model_type": regressor.model_type,
        "condition_mode": args.condition_mode,
        "num_images": len(rows),
        "dataset": args.dataset,
        "abs_rel": mean_absrel,
        "delta1": mean_d1,
        "roi_abs_rel": float(np.mean(roi_absrels)) if roi_absrels else None,
    }

    with (out_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    txt = (
        f"regressor_predepth eval\n"
        f"detail_model={args.detail_model}\n"
        f"regressor_dir={args.regressor_dir}\n"
        f"regressor_model_type={regressor.model_type}\n"
        f"in_channels={in_ch}\n"
        f"condition_mode={args.condition_mode}\n"
        f"abs_rel={mean_absrel:.6f}\n"
        f"delta1={mean_d1:.6f}\n"
        f"roi_abs_rel={summary['roi_abs_rel']}\n"
    )
    (out_dir / "eval_metrics-least_square_disparity.txt").write_text(txt, encoding="utf-8")
    logging.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
