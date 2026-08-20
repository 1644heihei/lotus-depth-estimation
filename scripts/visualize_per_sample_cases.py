#!/usr/bin/env python
"""Visualize best/worst per-sample eval cases (RGB, GT, predictions, pre-depth)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_regressor_predepth_nyuv2 import (
    eigen_valid_mask,
    predict_detail,
    score_prediction,
)
from evaluation.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from pipeline import LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.object_attention_condition import (
    ObjectAttentionEncoder,
    padded_object_condition_from_detections,
)
from utils.object_spatial_attention import install_object_spatial_attention_processors
from utils.object_depth_regressor import ObjectDepthRegressorBundle
from utils.object_detection_cache import load_detections
from utils.object_pre_depth import CoreDepthPredictor
from utils.object_pre_depth_regressor import build_pre_depth_for_rgb
from utils.seed_all import seed_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--comparison_csv",
        default="output/eval-object-attention-8k-6500/per_sample_comparison.csv",
    )
    p.add_argument(
        "--rgb_dir",
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument(
        "--detail_artifacts_dir",
        default="D:/lotus/data/nyuv2_detail_artifacts/test",
    )
    p.add_argument("--regressor_dir", default="output/object_depth_regressor_v4_mask")
    p.add_argument("--official_model", default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument(
        "--attention_model",
        default="output/train-lotus-d-object-attention-4ch-8k",
    )
    p.add_argument("--checkpoint_step", type=int, default=6500)
    p.add_argument("--output_dir", default="output/eval-object-attention-8k-6500/visualizations")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--processing_res", type=int, default=512)
    p.add_argument("--half_precision", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in (
                "official_abs",
                "off_abs",
                "on_abs",
                "delta_on_vs_official",
                "delta_on_vs_off",
            ):
                row[k] = float(row[k])
            row["n_det"] = int(row["n_det"])
            rows.append(row)
    return rows


def aligned_depth(pred_disp: np.ndarray, gt_depth: np.ndarray) -> np.ndarray:
    gt = gt_depth.astype(np.float64)
    pred = pred_disp.astype(np.float64)
    if pred.shape != gt.shape:
        pred = np.array(
            Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR),
            dtype=np.float64,
        )
    valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(*gt.shape[:2])
    gt_disp, gt_nn = depth2disparity(depth=gt, return_mask=True)
    valid_nn = valid & gt_nn & (pred > 0)
    disp_aligned, _, _ = align_depth_least_square(
        gt_arr=gt_disp,
        pred_arr=pred,
        valid_mask_arr=valid_nn,
        return_scale_shift=True,
    )
    return np.clip(disparity2depth(np.clip(disp_aligned, 1e-3, None)), 1e-3, 10.0)


def draw_detections(rgb_np: np.ndarray, dets, score_thr: float) -> np.ndarray:
    out = rgb_np.copy()
    for det in dets:
        if det.score < score_thr:
            continue
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name[:8]} {det.score:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out


def depth_panel(depth: np.ndarray, gt_depth: np.ndarray, title: str) -> Image.Image:
    valid = eigen_valid_mask(*gt_depth.shape[:2]) & np.isfinite(gt_depth) & (gt_depth > 1e-3)
    if valid.sum() > 0:
        vmin = float(np.percentile(gt_depth[valid], 5))
        vmax = float(np.percentile(gt_depth[valid], 95))
        depth_show = np.clip(depth, vmin, vmax)
    else:
        depth_show = depth
    img = colorize_depth_map(torch.from_numpy(depth_show.astype(np.float32)))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(255, 255, 255))
    return img


def add_title(img: Image.Image, title: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 18), fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(255, 255, 255))
    return out


def load_attention_pipe(model_dir: str, step: int, dtype, device):
    pipe = LotusDPipeline.from_pretrained(model_dir, torch_dtype=dtype).to(device)
    ckpt = Path(model_dir) / f"checkpoint-{step}"
    pipe.unet = UNet2DConditionModel.from_pretrained(
        ckpt, subfolder="unet", torch_dtype=dtype
    ).to(device)
    if pipe.object_condition_encoder is not None and (
        ckpt / "object_condition_encoder"
    ).is_dir():
        pipe.object_condition_encoder = ObjectAttentionEncoder.from_pretrained(
            ckpt, subfolder="object_condition_encoder", torch_dtype=dtype
        ).to(device)
    if pipe.object_condition_encoder is not None:
        install_object_spatial_attention_processors(pipe.unet)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def run_case(
    row: dict,
    rgb_dir: Path,
    detail_root: Path,
    official_pipe,
    attention_pipe,
    regressor,
    core_predictor,
    score_thr: float,
    processing_res: int,
    generator,
    out_path: Path,
    tag: str,
):
    rel = row["filename"]
    rgb_path = rgb_dir / rel
    depth_path = rgb_path.parent / rgb_path.name.replace("rgb_", "depth_", 1)
    rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
    gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
    h, w = rgb_np.shape[:2]

    dets = [d for d in load_detections(rgb_path, detail_root) if d.score >= score_thr]
    pre_depth, valid, _ = build_pre_depth_for_rgb(
        rgb_np, dets, regressor, core_predictor
    )
    object_condition = padded_object_condition_from_detections(
        dets, w, h, regressor, max_objects=16
    )

    official_disp = predict_detail(
        official_pipe, rgb_np, None, None, 999, processing_res, generator, None
    )
    attention_disp = predict_detail(
        attention_pipe,
        rgb_np,
        None,
        None,
        999,
        processing_res,
        generator,
        object_condition,
    )

    official_depth = aligned_depth(official_disp, gt)
    attention_depth = aligned_depth(attention_disp, gt)
    gt_depth = gt.copy()

    rgb_det = draw_detections(rgb_np, dets, score_thr)
    panels = [
        add_title(Image.fromarray(rgb_det), f"RGB + det ({len(dets)})"),
        depth_panel(gt_depth, gt_depth, "GT depth"),
        depth_panel(
            official_depth,
            gt_depth,
            f"Official abs={row['official_abs']:.3f}",
        ),
        depth_panel(
            attention_depth,
            gt_depth,
            f"AttnON abs={row['on_abs']:.3f}",
        ),
    ]
    if pre_depth is not None and valid is not None:
        pre_vis = pre_depth.copy()
        pre_vis[valid <= 0] = np.nan
        pre_vis = np.nan_to_num(pre_vis, nan=0.0)
        panels.append(
            depth_panel(pre_vis, gt_depth, "Regressor pre-depth")
        )

    tile_w = max(p.width for p in panels)
    tile_h = max(p.height for p in panels)
    cols = len(panels)
    header_h = 36
    canvas = Image.new("RGB", (tile_w * cols, tile_h + header_h), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    header = (
        f"[{tag}] {rel}  delta(on-official)={row['delta_on_vs_official']:+.3f}  "
        f"official={row['official_abs']:.3f} on={row['on_abs']:.3f} n_det={row['n_det']}"
    )
    draw.text((8, 8), header, fill=(255, 255, 255))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * tile_w, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    print(f"Saved {out_path}")


def case_stem(rel: str) -> str:
    return Path(rel).with_suffix("").as_posix().replace("/", "__")


def main():
    args = parse_args()
    seed_all(args.seed)
    comparison_csv = Path(args.comparison_csv)
    rows = load_rows(comparison_csv)
    best = sorted(rows, key=lambda r: r["delta_on_vs_official"])[: args.top_k]
    worst = sorted(rows, key=lambda r: r["delta_on_vs_official"], reverse=True)[
        : args.top_k
    ]

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    official_pipe = LotusDPipeline.from_pretrained(
        args.official_model, torch_dtype=dtype
    ).to(device)
    official_pipe.set_progress_bar_config(disable=True)
    attention_pipe = load_attention_pipe(
        args.attention_model, args.checkpoint_step, dtype, device
    )
    regressor = ObjectDepthRegressorBundle.load(args.regressor_dir, device=device)
    score_thr = float(regressor.config.get("detection_score_thr", 0.5))
    core_pipe = LotusDPipeline.from_pretrained(
        args.official_model, torch_dtype=dtype
    ).to(device)
    core_pipe.set_progress_bar_config(disable=True)
    core_predictor = CoreDepthPredictor(
        core_pipe, processing_res=args.processing_res, generator=generator
    )

    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_artifacts_dir)
    out_dir = Path(args.output_dir)

    for i, row in enumerate(best, 1):
        run_case(
            row,
            rgb_dir,
            detail_root,
            official_pipe,
            attention_pipe,
            regressor,
            core_predictor,
            score_thr,
            args.processing_res,
            generator,
            out_dir / "best" / f"{i:02d}_{case_stem(row['filename'])}.png",
            "BEST",
        )
    for i, row in enumerate(worst, 1):
        run_case(
            row,
            rgb_dir,
            detail_root,
            official_pipe,
            attention_pipe,
            regressor,
            core_predictor,
            score_thr,
            args.processing_res,
            generator,
            out_dir / "worst" / f"{i:02d}_{case_stem(row['filename'])}.png",
            "WORST",
        )


if __name__ == "__main__":
    main()
