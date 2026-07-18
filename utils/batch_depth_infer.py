"""Batch depth inference for Hypersim-processed training RGB images.

Only processes files matching --pattern (default rgb_cam_*.png).
Preserves directory layout under --output_dir.

Usage:
  python utils/batch_depth_infer.py \\
      --input_dir D:/lotus/data/hypersim_processed/train \\
      --output_dir D:/lotus/data/hypersim_pred_depth/train \\
      --pretrained_model_name_or_path jingheya/lotus-depth-d-v2-0-disparity \\
      --disparity --skip_existing
"""

import argparse
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers.utils import check_min_version
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.seed_all import seed_all

check_min_version("0.28.0.dev0")


def parse_args():
    p = argparse.ArgumentParser(description="Batch Lotus-D depth inference on training RGB images.")
    p.add_argument("--pretrained_model_name_or_path", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--pattern", type=str, default="rgb_cam_*.png")
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--half_precision", action="store_true", default=True)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--disparity", action="store_true", default=True)
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--save_npy", action="store_true", default=True)
    p.add_argument("--max_images", type=int, default=0, help="0 = all matching images.")
    return p.parse_args()


def iter_rgb_images(input_dir: Path, pattern: str, max_images: int = 0):
    files = sorted(input_dir.rglob(pattern))
    if max_images > 0:
        files = files[:max_images]
    return files


def output_paths(output_dir: Path, input_dir: Path, image_path: Path, save_npy: bool):
    rel = image_path.relative_to(input_dir)
    stem = rel.stem.replace("rgb_", "depth_pred_")
    vis_path = output_dir / "depth_vis" / rel.parent / f"{stem}.png"
    npy_path = output_dir / "depth_npy" / rel.parent / f"{stem}.npy" if save_npy else None
    check_paths = [vis_path] + ([npy_path] if npy_path else [])
    return vis_path, npy_path, check_paths


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.seed is not None:
        seed_all(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    images = iter_rgb_images(input_dir, args.pattern, args.max_images)
    logging.info("Found %d RGB images under %s (pattern=%s)", len(images), input_dir, args.pattern)

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = LotusDPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    skipped = 0
    done = 0

    with torch.no_grad():
        for image_path in tqdm(images, desc="Depth infer"):
            vis_path, npy_path, check_paths = output_paths(output_dir, input_dir, image_path, args.save_npy)
            if args.skip_existing and all(p.exists() for p in check_paths):
                skipped += 1
                continue

            rgb_np = np.array(Image.open(image_path).convert("RGB"))
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

            depth = pred.mean(axis=-1).astype(np.float32)
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            colorize_depth_map(depth, reverse_color=args.disparity).save(vis_path)
            if npy_path is not None:
                npy_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(npy_path, depth)
            done += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

    logging.info("Done. inferred=%d skipped=%d output=%s", done, skipped, output_dir)


if __name__ == "__main__":
    main()
