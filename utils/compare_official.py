"""Compare fine-tuned semantic Lotus-D vs official Lotus-D on the same images."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import LotusDPipeline
from utils.image_utils import colorize_depth_map, concatenate_images
from utils.midtrain_eval import load_gt_depth, pick_samples, resize_depth
from utils.semantic_fusion import enable_vae_early_fusion
from utils.semantic_mask_utils import load_mask_for_image, mask_to_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--official_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--ours_checkpoint", type=str, default="output/train-lotus-d-depth-semantic-mask-bsz8/checkpoint-20000")
    p.add_argument("--ours_base", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--mask_dir", type=str, default=r"D:\lotus\data\hypersim_sem_masks")
    p.add_argument("--data_dir", type=str, default=r"D:\lotus\data\hypersim_processed\train")
    p.add_argument("--output_dir", type=str, default="output/compare_official_vs_ours")
    p.add_argument("--processing_res", type=int, default=576)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--timestep", type=int, default=999)
    return p.parse_args()


def load_pipeline(model_path, unet_path=None, early_fusion=False, device="cuda"):
    kwargs = dict(torch_dtype=torch.float16, safety_checker=None)
    if unet_path:
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
        pipeline = LotusDPipeline.from_pretrained(model_path, unet=unet, **kwargs)
    else:
        pipeline = LotusDPipeline.from_pretrained(model_path, **kwargs)
    if early_fusion:
        enable_vae_early_fusion(pipeline.vae)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@torch.no_grad()
def predict_depth(pipeline, rgb_path, device, args, mask_dir=None, early_fusion=False):
    rgb = Image.open(rgb_path).convert("RGB")
    arr = np.array(rgb).astype(np.float32)
    t = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    t = t.to(device)

    sem = None
    if early_fusion and mask_dir:
        mask_np = load_mask_for_image(rgb_path, mask_dir)
        sem = mask_to_tensor(mask_np, device=device) if mask_np is not None else None

    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).to(device)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    with torch.autocast(device.type):
        pred = pipeline(
            rgb_in=t,
            task_emb=task_emb,
            prompt="",
            timesteps=[args.timestep],
            output_type="np",
            semantic_mask=sem,
            semantic_fusion_mode="early" if early_fusion else "auto",
            processing_res=args.processing_res,
            match_input_res=True,
        ).images[0]
    return rgb, pred.mean(axis=-1)


def si_abs_rel(pred, gt, valid):
  if not valid.any():
      return float("nan")
  p = pred[valid].astype(np.float64)
  g = gt[valid].astype(np.float64)
  scale = np.median(g) / (np.median(p) + 1e-8)
  p = p * scale
  return float(np.mean(np.abs(p - g) / g))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[compare] loading official Lotus-D ...")
    official = load_pipeline(args.official_model, early_fusion=False, device=device)

    print("[compare] loading ours (semantic early fusion) ...")
    ours = load_pipeline(
        args.ours_base,
        unet_path=os.path.join(args.ours_checkpoint, "unet"),
        early_fusion=True,
        device=device,
    )

    samples = pick_samples(args.data_dir, n=args.num_samples)
    rows_official, rows_ours, rows_gt, rows_rgb = [], [], [], []
    metrics = []

    for rgb_path, depth_path in samples:
        gt = load_gt_depth(depth_path)
        _, pred_off = predict_depth(official, rgb_path, device, args)
        rgb, pred_ours = predict_depth(ours, rgb_path, device, args, mask_dir=args.mask_dir, early_fusion=True)
        gt_rs = resize_depth(gt, pred_off.shape)

        valid = (gt_rs > 1e-3) & np.isfinite(pred_off) & np.isfinite(pred_ours) & np.isfinite(gt_rs)
        m_off = si_abs_rel(pred_off, gt_rs, valid)
        m_ours = si_abs_rel(pred_ours, gt_rs, valid)
        metrics.append((Path(rgb_path).name, m_off, m_ours))

        rows_rgb.append(rgb.resize((pred_off.shape[1], pred_off.shape[0])))
        rows_official.append(colorize_depth_map(pred_off, reverse_color=True))
        rows_ours.append(colorize_depth_map(pred_ours, reverse_color=True))
        rows_gt.append(colorize_depth_map(gt_rs, reverse_color=True))

    grid = concatenate_images(rows_rgb, rows_official, rows_ours, rows_gt)
    vis_path = os.path.join(args.output_dir, "compare_vis.png")
    grid.save(vis_path)

    report = os.path.join(args.output_dir, "report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("Lotus-D comparison on Hypersim train samples\n")
        f.write(f"official: {args.official_model}\n")
        f.write(f"ours: {args.ours_checkpoint} (+ YOLO mask early fusion)\n")
        f.write(f"visual: {vis_path}\n\n")
        f.write("Rows: RGB | Official Lotus-D | Ours (semantic) | GT\n\n")
        f.write("scale-invariant abs_rel (rough, lower=better):\n")
        f.write(f"{'image':40s} {'official':>10s} {'ours':>10s}\n")
        for name, mo, mu in metrics:
            f.write(f"{name:40s} {mo:10.4f} {mu:10.4f}\n")
        off_mean = np.nanmean([m[1] for m in metrics])
        ours_mean = np.nanmean([m[2] for m in metrics])
        f.write(f"\nmean official={off_mean:.4f}  ours={ours_mean:.4f}\n")
        f.write("\nNotes:\n")
        f.write("- Official: SD2 + VKITTI/Hypersim, disparity, 3ch, no mask\n")
        f.write("- Ours: SD1.5 + Hypersim only, trunc_disparity, 4ch early fusion + YOLO mask\n")
        f.write("- Not apples-to-apples training; Hypersim samples favor our fine-tune distribution.\n")

    print(open(report, encoding="utf-8").read())


if __name__ == "__main__":
    main()
