#!/usr/bin/env python
"""Does a contour survive Lotus's VAE, or is its precision the first thing thrown away?

Conditioning on contours during training is worth considering because training can learn to
tolerate a contour that is 2px off, which the post-hoc operation cannot - it trusts the
position and needs 1.71px. But the obvious way to feed one, as an extra input channel,
sends it through the VAE encoder and its 8x downsample first.

That matters more here than it usually would: what a contour contributes is position, at a
tolerance of one or two pixels, and 8x downsampling is exactly an operation that discards
detail below its cell size. If the contour cannot come back out of the VAE, the channel
carries the one thing it was added for straight into the bottleneck.

Measured by encoding a contour map, decoding it, re-extracting the contour, and asking how
far the recovered line sits from where it went in - the same displacement statistic the
tolerance curve is drawn against, so the answer lands on the same axis as the 1.71px
budget.

Two references make the number readable. A depth map through the same round trip says how
much of the loss is the VAE being lossy in general rather than contours specifically, and
plain 8x down/upsampling says how much is the downsample alone rather than the learned
codec.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path, align_to_gt, load_or_build_masks
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs


def parse_args():
    p = argparse.ArgumentParser(description="Contour survival through the VAE.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/sam_seg")
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--output_dir", type=str, default="output/eval_vae_contour_survival")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--max_images", type=int, default=60)
    return p.parse_args()


@torch.no_grad()
def vae_round_trip(vae, img01, device):
    """[H,W] in 0..1 -> 3 channels -> VAE encode/decode -> [H,W] in 0..1."""
    x = torch.from_numpy(img01).float()[None, None].repeat(1, 3, 1, 1).to(device)
    x = x * 2.0 - 1.0
    lat = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
    out = vae.decode(lat / vae.config.scaling_factor).sample
    return ((out[0].mean(0).clamp(-1, 1) + 1.0) / 2.0).cpu().numpy()


def pad_to_multiple(a, m=8):
    h, w = a.shape
    return np.pad(a, ((0, (-h) % m), (0, (-w) % m)), mode="edge"), h, w


def displacement(a: np.ndarray, b: np.ndarray) -> float:
    """Median distance from each pixel of `a` to the nearest pixel of `b`."""
    if not a.any() or not b.any():
        return float("nan")
    return float(np.median(ndi.distance_transform_edt(~b)[a]))


def main():
    args = parse_args()
    from diffusers import AutoencoderKL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.core_model, subfolder="vae",
                                        torch_dtype=torch.float32).to(device).eval()
    print(f"VAE downsample factor: {2 ** (len(vae.config.block_out_channels) - 1)}x")

    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = {k: [] for k in ["contour_vae", "contour_naive8", "depth_disc_vae",
                           "contour_recall_vae", "contour_frac_in", "contour_frac_out"]}

    for rgb_path, depth_path in tqdm(list_nyu_pairs(rgb_dir)[: args.max_images], desc="vae"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        seg = list(load_or_build_masks(rgb_path, rgb_dir, Path(args.mask_cache_dir), None,
                                       np.empty((h, w, 3), np.uint8), 0.5))
        if not seg:
            continue
        u = np.any(np.stack(seg), axis=0).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        cont = (cv2.dilate(u, k).astype(bool) & ~cv2.erode(u, k).astype(bool)) & valid
        if not cont.any():
            continue

        padded, H, W = pad_to_multiple(cont.astype(np.float64))
        rec = vae_round_trip(vae, padded, device)[:H, :W]
        # threshold at the midpoint: the input was exactly 0 or 1
        rec_c = rec > 0.5
        acc["contour_frac_in"].append(float(cont.mean()))
        acc["contour_frac_out"].append(float(rec_c.mean()))
        acc["contour_vae"].append(displacement(cont, rec_c))
        acc["contour_recall_vae"].append(displacement(rec_c, cont))

        # reference 1: the downsample alone, no learned codec
        small = cv2.resize(cont.astype(np.float32), (max(W // 8, 1), max(H // 8, 1)),
                           interpolation=cv2.INTER_AREA)
        naive = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        acc["contour_naive8"].append(displacement(cont, naive))

        # reference 2: a depth map through the same trip, scored on its discontinuities
        pp = _cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        if pp.is_file():
            base = align_to_gt(np.load(pp).astype(np.float64), gt, valid)
            if base is not None:
                lo, hi = np.percentile(base[valid], [1, 99])
                norm = np.clip((base - lo) / max(hi - lo, 1e-9), 0, 1)
                padded, H2, W2 = pad_to_multiple(norm)
                back = vae_round_trip(vae, padded, device)[:H2, :W2] * (hi - lo) + lo
                acc["depth_disc_vae"].append(displacement(
                    discontinuities(base, valid, args.t),
                    discontinuities(back, valid, args.t)))

    summary = {"n_images": len(acc["contour_vae"]),
               **{k: float(np.nanmedian(v)) if v else None for k, v in acc.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nContour survival through the VAE   n={summary['n_images']}")
    print(f"\n{'':<44}{'median px':>11}")
    print("-" * 55)
    print(f"{'contour in -> nearest recovered contour':<44}"
          f"{summary['contour_vae']:>11.2f}")
    print(f"{'recovered contour -> nearest input contour':<44}"
          f"{summary['contour_recall_vae']:>11.2f}")
    print(f"{'  reference: plain 8x down/up, no codec':<44}"
          f"{summary['contour_naive8']:>11.2f}")
    if summary["depth_disc_vae"] is not None:
        print(f"{'  reference: a DEPTH map through the same trip':<44}"
              f"{summary['depth_disc_vae']:>11.2f}")
    print(f"\ncontour pixels: {summary['contour_frac_in']*100:.2f}% in, "
          f"{summary['contour_frac_out']*100:.2f}% out")
    print("\nThe sharpening budget is 1.71px. A contour arriving further than that has "
          "already\nlost what it was added for before the UNet sees it.")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
