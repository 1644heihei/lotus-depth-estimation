"""Quick mid-training sanity check: inference + loss debug."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import LotusDPipeline
from utils.image_utils import colorize_depth_map, concatenate_images
from utils.semantic_fusion import enable_vae_early_fusion
from utils.semantic_mask_utils import load_mask_for_image, mask_to_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--mask_dir", type=str, default=r"D:\lotus\data\hypersim_sem_masks")
    p.add_argument("--data_dir", type=str, default=r"D:\lotus\data\hypersim_processed\train")
    p.add_argument("--output_dir", type=str, default="output/midtrain_check")
    p.add_argument("--processing_res", type=int, default=576)
    p.add_argument("--timestep", type=int, default=999)
    return p.parse_args()


def load_gt_depth(depth_path: str) -> np.ndarray:
    depth = np.array(Image.open(depth_path))
    if depth.dtype == np.uint16 and depth.max() > 1000:
        depth = depth.astype(np.float32) / 100.0
    else:
        depth = depth.astype(np.float32)
    return np.clip(depth, 1e-4, None)


def resize_depth(depth: np.ndarray, hw) -> np.ndarray:
    t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=hw, mode="nearest")
    return t.squeeze().numpy()


def abs_rel(pred, gt, mask):
    pred = pred[mask]
    gt = gt[mask]
    return float(np.mean(np.abs(pred - gt) / gt))


def pick_samples(data_dir: str, n=4):
    root = Path(data_dir)
    scenes = sorted([p for p in root.iterdir() if p.is_dir()])
    samples = []
    for scene in scenes[:: max(1, len(scenes) // n)][:n]:
        rgbs = sorted(scene.glob("rgb_cam_*.png"))
        if rgbs:
            rgb = rgbs[len(rgbs) // 2]
            depth = Path(str(rgb).replace("rgb_cam_", "depth_plane_cam_"))
            if depth.exists():
                samples.append((str(rgb), str(depth)))
    return samples


def run_inference(pipeline, rgb_path, mask_dir, device, args):
    rgb = Image.open(rgb_path).convert("RGB")
    arr = np.array(rgb).astype(np.float32)
    t = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    t = t.to(device)

    mask_np = load_mask_for_image(rgb_path, mask_dir)
    sem = mask_to_tensor(mask_np, device=device) if mask_np is not None else None

    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).to(device)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    with torch.autocast(device.type), torch.no_grad():
        pred = pipeline(
            rgb_in=t,
            task_emb=task_emb,
            prompt="",
            timesteps=[args.timestep],
            output_type="np",
            semantic_mask=sem,
            semantic_fusion_mode="early",
            processing_res=args.processing_res,
            match_input_res=True,
        ).images[0]
    pred_depth = pred.mean(axis=-1)
    return rgb, mask_np, pred_depth


def debug_loss_one_batch(base_model, unet_path, data_dir, mask_dir, device):
    from transformers import CLIPTokenizer, CLIPTextModel
    from utils.semantic_fusion import concat_rgb_and_mask, append_constant_channel

    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device)
    enable_vae_early_fusion(vae)
    unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device)
    vae.eval(); unet.eval(); text_encoder.eval()

    rgb_path, depth_path = pick_samples(data_dir, n=1)[0]
    rgb = Image.open(rgb_path).convert("RGB")
    w, h = rgb.size
    if h > w:
        new_w, new_h = 576, int(576 * h / w)
    else:
        new_h, new_w = 576, int(576 * w / h)
    rgb = rgb.resize((new_w, new_h), Image.BILINEAR)
    gt = load_gt_depth(depth_path)
    gt = resize_depth(gt, (new_h, new_w))

    px = torch.tensor(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    px = px.to(device)
    mask_np = load_mask_for_image(rgb_path, mask_dir)
    sem = mask_to_tensor(mask_np, device=device) if mask_np is not None else None
    if sem is not None:
        sem = F.interpolate(sem, size=(new_h, new_w), mode="bilinear", align_corners=False)

    rgb4 = concat_rgb_and_mask(torch.cat([px, px], dim=0), torch.cat([sem, sem], dim=0) if sem is not None else None)
    latents = vae.encode(rgb4).latent_dist.sample() * vae.config.scaling_factor

    dep = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
    dep3 = dep.repeat(1, 3, 1, 1)
    dep3 = (dep3 / dep3.max().clamp(min=1e-3)) * 2.0 - 1.0
    dep3 = dep3.to(device)
    target_concat = torch.cat(
        [append_constant_channel(dep3, value=0.0), append_constant_channel(px, value=0.0)],
        dim=0,
    )
    target_latents = vae.encode(target_concat).latent_dist.sample() * vae.config.scaling_factor

    bsz = target_latents.shape[0]
    timesteps = torch.tensor([999], device=device).repeat(bsz).long()
    text_inputs = tokenizer("", padding="do_not_pad", max_length=tokenizer.model_max_length, return_tensors="pt")
    enc = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0].repeat(bsz, 1, 1)

    task_anno = torch.cat([torch.sin(torch.tensor([[1., 0.]])), torch.cos(torch.tensor([[1., 0.]]))], dim=-1).to(device)
    task_rgb = torch.cat([torch.sin(torch.tensor([[0., 1.]])), torch.cos(torch.tensor([[0., 1.]]))], dim=-1).to(device)
    task_emb = torch.cat([task_anno, task_rgb], dim=0)

    pred = unet(latents, timesteps, enc, return_dict=False, class_labels=task_emb)[0]
    anno_loss = F.mse_loss(pred[:1].float(), target_latents[:1].float())
    rgb_loss = F.mse_loss(pred[1:].float(), target_latents[1:].float())
    return {
        "rgb": rgb_path,
        "pred_finite": bool(torch.isfinite(pred).all()),
        "target_finite": bool(torch.isfinite(target_latents).all()),
        "latents_finite": bool(torch.isfinite(latents).all()),
        "anno_loss": float(anno_loss),
        "rgb_loss": float(rgb_loss),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_path = os.path.join(args.checkpoint, "unet")
    pipeline = LotusDPipeline.from_pretrained(
        args.base_model,
        unet=UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16),
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    enable_vae_early_fusion(pipeline.vae)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    samples = pick_samples(args.data_dir, n=4)
    rows_rgb, rows_mask, rows_pred, rows_gt = [], [], [], []
    metrics = []

    for rgb_path, depth_path in samples:
        rgb, mask_np, pred = run_inference(pipeline, rgb_path, args.mask_dir, device, args)
        gt = load_gt_depth(depth_path)
        gt_rs = resize_depth(gt, pred.shape)
        valid = (gt_rs > 1e-3) & np.isfinite(pred) & np.isfinite(gt_rs)
        m = abs_rel(pred, gt_rs, valid) if valid.any() else float("nan")
        metrics.append((Path(rgb_path).name, m))

        mask_vis = Image.fromarray((mask_np * 255).astype(np.uint8) if mask_np is not None else np.zeros(pred.shape, np.uint8))
        rows_rgb.append(rgb.resize((pred.shape[1], pred.shape[0])))
        rows_mask.append(mask_vis.convert("RGB"))
        rows_pred.append(colorize_depth_map(pred, reverse_color=True))
        rows_gt.append(colorize_depth_map(gt_rs, reverse_color=True))

    grid = concatenate_images(rows_rgb, rows_mask, rows_pred, rows_gt)
    out_img = os.path.join(args.output_dir, "checkpoint_vis.png")
    grid.save(out_img)

    dbg = debug_loss_one_batch(args.base_model, unet_path, args.data_dir, args.mask_dir, device)

    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"visual: {out_img}\n\n")
        f.write("abs_rel (lower is better, rough):\n")
        for name, m in metrics:
            f.write(f"  {name}: {m:.4f}\n")
        f.write("\nloss_debug:\n")
        for k, v in dbg.items():
            f.write(f"  {k}: {v}\n")

    print(open(report_path, encoding="utf-8").read())


if __name__ == "__main__":
    main()
