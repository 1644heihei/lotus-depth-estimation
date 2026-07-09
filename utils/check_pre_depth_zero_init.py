"""Sanity check: expanded UNet matches original when extra inputs are zero.

Usage:
  python utils/check_pre_depth_zero_init.py \
      --pretrained_model_name_or_path jingheya/lotus-depth-d-v2-0-disparity
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.pre_depth_fusion import expand_unet_conv_in


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--tol", type=float, default=1e-6)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        class_embed_type="projection",
        projection_class_embeddings_input_dim=4,
        low_cpu_mem_usage=False,
        device_map=None,
    ).to(device=device, dtype=dtype)
    ext = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        class_embed_type="projection",
        projection_class_embeddings_input_dim=4,
        low_cpu_mem_usage=False,
        device_map=None,
    ).to(device=device, dtype=dtype)
    expand_unet_conv_in(ext, extra_in_channels=5, zero_init=True)

    b, c, h, w = 2, 4, 64, 80
    x = torch.randn((b, c, h, w), device=device, dtype=dtype)
    x_ext = torch.cat([x, torch.zeros((b, 5, h, w), device=device, dtype=dtype)], dim=1)
    t = torch.tensor([999, 999], device=device).long()
    emb = torch.randn((b, 77, base.config.cross_attention_dim), device=device, dtype=dtype)
    class_labels = torch.randn((b, 4), device=device, dtype=dtype)

    y_base = base(x, t, encoder_hidden_states=emb, class_labels=class_labels, return_dict=False)[0]
    y_ext = ext(x_ext, t, encoder_hidden_states=emb, class_labels=class_labels, return_dict=False)[0]
    max_err = (y_base - y_ext).abs().max().item()
    print(f"max_abs_error={max_err:.8f}")
    if max_err > args.tol:
        raise SystemExit(f"FAILED: max_abs_error {max_err:.8f} > tol {args.tol}")
    print("PASS")


if __name__ == "__main__":
    main()

