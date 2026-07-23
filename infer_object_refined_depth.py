#!/usr/bin/env python
"""End-to-end inference: core model + YOLO + detail model (Approach A)."""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from pipeline import LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.object_condition import class_map_to_tensor, rasterize_class_map
from utils.object_detection_cache import load_yolo_model, run_yolo_detections
from utils.object_pre_depth import CoreDepthPredictor, disparity_pred_to_norm
from utils.seed_all import seed_all


def parse_args():
    p = argparse.ArgumentParser(description="Object-refined depth inference (core + YOLO + detail).")
    p.add_argument("--core_model", type=str, required=True)
    p.add_argument("--detail_model", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--yolo_model", type=str, default="yolov8n-seg.pt")
    p.add_argument("--yolo_score_thr", type=float, default=0.25)
    p.add_argument("--roi_expand_ratio", type=float, default=0.25)
    p.add_argument("--align_mode", type=str, default="lstsq", choices=["lstsq", "none"])
    p.add_argument("--disparity", action="store_true")
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0, help="If >0, only process first N images.")
    return p.parse_args()


def predict_with_pre_depth(pipe, rgb_np, pre_depth_norm, valid_mask, class_map, args, generator):
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = image / 127.5 - 1.0
    image = image.to(device)

    pre_t = torch.from_numpy(pre_depth_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pre_t = pre_t.repeat(1, 3, 1, 1).to(device=device, dtype=image.dtype)
    valid_t = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device=device, dtype=image.dtype)

    extra_channels = pipe.unet.config.in_channels - 4
    class_t = None
    if extra_channels == 9 and class_map is not None:
        class_norm = class_map_to_tensor(class_map)
        class_t = torch.from_numpy(class_norm).unsqueeze(0).unsqueeze(0).to(device=device, dtype=image.dtype)

    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type=device.type)

    pipe_kwargs = dict(
        rgb_in=image,
        pre_depth=pre_t,
        pre_depth_valid_mask=valid_t,
        prompt="",
        num_inference_steps=1,
        generator=generator,
        output_type="np",
        timesteps=[args.timestep],
        task_emb=task_emb,
        processing_res=args.processing_res,
        match_input_res=True,
    )
    if class_t is not None:
        pipe_kwargs["class_map"] = class_t

    with autocast_ctx:
        pred = pipe(**pipe_kwargs).images[0]
    return pred.mean(axis=-1).astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    core_pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=dtype).to(device)
    detail_pipe = LotusDPipeline.from_pretrained(args.detail_model, torch_dtype=dtype).to(device)
    core_pipe.set_progress_bar_config(disable=True)
    detail_pipe.set_progress_bar_config(disable=True)

    yolo = load_yolo_model(args.yolo_model)
    core_predictor = CoreDepthPredictor(core_pipe, timestep=args.timestep, processing_res=args.processing_res, generator=generator)

    images = sorted(list(Path(args.input_dir).rglob("*.png")) + list(Path(args.input_dir).rglob("*.jpg")))
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    out_depth = Path(args.output_dir) / "depth"
    out_vis = Path(args.output_dir) / "depth_vis"
    out_pre = Path(args.output_dir) / "pre_depth"
    for d in (out_depth, out_vis, out_pre):
        d.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images, desc="object_refined_depth"):
        rgb_np = np.array(Image.open(image_path).convert("RGB"))
        detections = run_yolo_detections(rgb_np, model=yolo, score_thr=args.yolo_score_thr)
        pre_depth_norm, valid_mask, _ = core_predictor.build_pre_depth(
            rgb_np,
            detections,
            roi_expand_ratio=args.roi_expand_ratio,
            align_mode=args.align_mode,
        )
        h, w = rgb_np.shape[:2]
        class_map = rasterize_class_map(detections, h, w)
        refined = predict_with_pre_depth(
            detail_pipe, rgb_np, pre_depth_norm, valid_mask, class_map, args, generator
        )

        stem = image_path.stem
        np.save(out_depth / f"{stem}.npy", refined)
        np.save(out_pre / f"{stem}.npy", pre_depth_norm)
        colorize_depth_map(refined, reverse_color=args.disparity).save(out_vis / f"{stem}.png")

    logging.info("Done. output=%s", args.output_dir)


if __name__ == "__main__":
    main()
