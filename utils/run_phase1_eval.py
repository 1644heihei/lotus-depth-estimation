"""Phase 1 evaluation: consistent ROI fusion vs. global-only baseline on NYUv2.

Compares three conditions with a single pass of model inference
(crop predictions are reused across fusion modes):

  1. global : global prediction only (official Lotus-D baseline)
  2. paste  : crop predictions pasted without alignment (previous approach)
  3. lstsq  : crop predictions scale-shift aligned to the global prediction

Metrics per condition: overall / ROI / boundary-band abs_rel & delta1,
plus depth boundary F-score. Predictions are aligned to GT in disparity
space (least squares), following the official Lotus evaluation protocol.

Usage:
  python utils/run_phase1_eval.py \
      --mask_dir data/nyu_sem_masks \
      --output_dir output/phase1_roi_fusion \
      [--max_images 20]
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.dataset_depth import DatasetMode, get_dataset
from evaluation.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from pipeline import LotusDPipeline
from utils.boundary_metrics import boundary_f1
from utils.roi_fusion import fuse_roi_depth
from utils.semantic_mask_utils import connected_component_boxes, expand_box, load_mask_for_image

CONDITIONS = ("global", "paste", "lstsq")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 ROI fusion evaluation on NYUv2.")
    p.add_argument("--pretrained_model_name_or_path", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--dataset_config", type=str, default="datasets/eval/depth/configs/data_nyu_test.yaml")
    p.add_argument("--base_data_dir", type=str, default="datasets/eval/depth")
    p.add_argument("--mask_dir", type=str, required=True, help="Root of YOLO object masks (mirrors test/scene layout).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--processing_res", type=int, default=None, help="None = model default (768).")
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--half_precision", action="store_true", default=True)
    p.add_argument("--roi_min_area", type=int, default=500)
    p.add_argument("--roi_expand_ratio", type=float, default=0.25)
    p.add_argument("--blend_blur_ksize", type=int, default=31)
    p.add_argument("--boundary_width", type=int, default=8)
    p.add_argument("--boundary_tolerance_px", type=int, default=3)
    p.add_argument("--max_images", type=int, default=0, help="0 = all test images.")
    p.add_argument("--save_vis_every", type=int, default=0, help="Save fused visualization every N samples (0 = off).")
    return p.parse_args()


def predict_disparity(pipe, rgb_np: np.ndarray, args, generator) -> np.ndarray:
    """Run Lotus-D on a uint8 RGB array, return [H, W] disparity prediction."""
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = (image / 127.5 - 1.0).to(device)
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    autocast_ctx = nullcontext() if device.type == "mps" else torch.autocast(device_type=device.type)
    with autocast_ctx:
        pred = pipe(
            rgb_in=image,
            prompt="",
            num_inference_steps=1,
            generator=generator,
            output_type="np",
            timesteps=[args.timestep],
            task_emb=task_emb,
            processing_res=args.processing_res,
            match_input_res=True,
        ).images[0]
    return pred.mean(axis=-1).astype(np.float32)


def align_pred_to_gt_depth(pred_disparity: np.ndarray, gt_depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Least-squares align prediction to GT in disparity space, return metric depth."""
    gt_disparity, gt_non_neg = depth2disparity(gt_depth, return_mask=True)
    vm = valid_mask & gt_non_neg & (pred_disparity > 0)
    aligned, _, _ = align_depth_least_square(
        gt_arr=gt_disparity,
        pred_arr=pred_disparity,
        valid_mask_arr=vm,
        return_scale_shift=True,
    )
    aligned = np.clip(aligned, a_min=1e-3, a_max=None)
    return disparity2depth(aligned)


def region_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, region_mask: np.ndarray) -> dict:
    n = max(int(region_mask.sum()), 1)
    gt_safe = np.clip(gt_depth, 1e-6, None)
    absrel = float((np.abs(pred_depth - gt_depth) / gt_safe)[region_mask].sum() / n)
    ratio = np.maximum(pred_depth / gt_safe, gt_depth / np.clip(pred_depth, 1e-6, None))
    delta1 = float((ratio[region_mask] < 1.25).mean()) if region_mask.any() else 0.0
    return {"absrel": absrel, "delta1": delta1}


def boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    k = np.ones((max(1, width), max(1, width)), dtype=np.uint8)
    binary = (mask > 0.5).astype(np.uint8)
    return ((cv2.dilate(binary, k) - cv2.erode(binary, k)) > 0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (args.half_precision and device.type == "cuda") else torch.float32
    pipe = LotusDPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    cfg_data = OmegaConf.load(args.dataset_config)
    dataset = get_dataset(cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.EVAL)

    num_samples = len(dataset)
    if args.max_images > 0:
        num_samples = min(num_samples, args.max_images)

    rows = []
    sums = defaultdict(float)
    counts = defaultdict(int)
    num_with_roi = 0

    for idx in tqdm(range(num_samples), desc="Phase1 eval"):
        data = dataset[idx]
        rgb_np = data["rgb_int"].numpy().transpose(1, 2, 0).astype(np.uint8)
        gt_depth = data["depth_raw_linear"].squeeze().numpy()
        valid_mask = data["valid_mask_raw"].squeeze().numpy().astype(bool)
        rel_path = data["rgb_relative_path"]
        h, w = gt_depth.shape

        roi_mask = load_mask_for_image(rel_path, args.mask_dir)
        if roi_mask is None:
            roi_mask = np.zeros((h, w), dtype=np.float32)
        if roi_mask.shape != (h, w):
            roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        global_disp = predict_disparity(pipe, rgb_np, args, generator)

        boxes = connected_component_boxes(roi_mask, threshold=0.5, min_area=args.roi_min_area)
        crop_preds = []
        for box in boxes:
            x1, y1, x2, y2 = expand_box(box, (h, w), expand_ratio=args.roi_expand_ratio)
            crop = rgb_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_preds.append(((x1, y1, x2, y2), predict_disparity(pipe, crop, args, generator)))

        preds_disp = {"global": global_disp}
        for mode, cond in (("none", "paste"), ("lstsq", "lstsq")):
            fused, _ = fuse_roi_depth(
                global_depth=global_disp,
                crop_preds=crop_preds,
                roi_mask=roi_mask,
                align_mode=mode,
                blend_blur_ksize=args.blend_blur_ksize,
            )
            preds_disp[cond] = fused

        has_roi = len(crop_preds) > 0 and (roi_mask > 0.5).any()
        if has_roi:
            num_with_roi += 1
        roi_region = valid_mask & (roi_mask > 0.5)
        boundary_region = valid_mask & boundary_band(roi_mask, args.boundary_width)

        row = {"image": rel_path.replace("/", "_"), "num_rois": len(crop_preds)}
        for cond in CONDITIONS:
            pred_depth = align_pred_to_gt_depth(preds_disp[cond], gt_depth, valid_mask)
            pred_depth = np.clip(pred_depth, dataset.min_depth, dataset.max_depth)

            overall = region_metrics(pred_depth, gt_depth, valid_mask)
            row[f"{cond}_absrel"] = overall["absrel"]
            row[f"{cond}_delta1"] = overall["delta1"]
            _, _, bf1 = boundary_f1(pred_depth, gt_depth, valid_mask, tolerance_px=args.boundary_tolerance_px)
            row[f"{cond}_bf1"] = bf1
            sums[f"{cond}_absrel"] += overall["absrel"]
            sums[f"{cond}_delta1"] += overall["delta1"]
            sums[f"{cond}_bf1"] += bf1
            counts[cond] += 1

            if has_roi and roi_region.any():
                roi_m = region_metrics(pred_depth, gt_depth, roi_region)
                row[f"{cond}_roi_absrel"] = roi_m["absrel"]
                sums[f"{cond}_roi_absrel"] += roi_m["absrel"]
                counts[f"{cond}_roi"] += 1
            if has_roi and boundary_region.any():
                bnd_m = region_metrics(pred_depth, gt_depth, boundary_region)
                row[f"{cond}_bnd_absrel"] = bnd_m["absrel"]
                sums[f"{cond}_bnd_absrel"] += bnd_m["absrel"]
                counts[f"{cond}_bnd"] += 1

        rows.append(row)

        if args.save_vis_every > 0 and idx % args.save_vis_every == 0:
            from utils.image_utils import colorize_depth_map

            vis_dir = os.path.join(args.output_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            panels = [colorize_depth_map(preds_disp[c], reverse_color=True) for c in CONDITIONS]
            total_w = sum(p.width for p in panels)
            from PIL import Image

            canvas = Image.new("RGB", (total_w, panels[0].height))
            x = 0
            for p in panels:
                canvas.paste(p, (x, 0))
                x += p.width
            canvas.save(os.path.join(vis_dir, f"{row['image']}.png"))

    # ---- write outputs ----
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(os.path.join(args.output_dir, "per_sample.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"num_samples": len(rows), "num_with_roi": num_with_roi, "conditions": {}}
    for cond in CONDITIONS:
        entry = {
            "absrel": sums[f"{cond}_absrel"] / max(counts[cond], 1),
            "delta1": sums[f"{cond}_delta1"] / max(counts[cond], 1),
            "boundary_f1": sums[f"{cond}_bf1"] / max(counts[cond], 1),
        }
        if counts[f"{cond}_roi"] > 0:
            entry["roi_absrel"] = sums[f"{cond}_roi_absrel"] / counts[f"{cond}_roi"]
        if counts[f"{cond}_bnd"] > 0:
            entry["boundary_band_absrel"] = sums[f"{cond}_bnd_absrel"] / counts[f"{cond}_bnd"]
        summary["conditions"][cond] = entry

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
