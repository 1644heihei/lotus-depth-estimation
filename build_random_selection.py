r"""Build the two skill-free selection controls at the learned selector's own budget.

These controls answer "is the gain just sparsity?", so they have to spend the same
budget as the selector. The masks used in the first round of measurements kept 4.06% and
4.14% of the contour against the selector's 6.00%, and the script that produced them was
not kept - the same failure as the hand-crafted threshold control. This one takes the
budget from a reference mask cache frame by frame, so the match is exact rather than
nominal, and takes a seed so the draw can be repeated.

Two controls:

  pixels  uniform over contour pixels. Destroys the contour's structure as well as
          choosing badly, which is why it is the harsher of the two.
  blocks  uniform over square tiles, accepting whole tiles until the budget is met.
          Keeps contour fragments contiguous, so it isolates choosing badly from
          scattering the selection.

Writes contour planes in the same packed format as build_sam_masks.py, so
`eval_yolo_contour_sharpening.py --masks_are_contours` scores them unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from contour_selector.paths import cache_path
from eval_object_oracle_ceiling import load_or_build_masks
from eval_regressor_predepth_nyuv2 import eigen_valid_mask
from utils.dataset_paths import resolve
from utils.eval_frames import add_dataset_args, depth_scale, list_frames, valid_mask


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("pixels", "blocks"), required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--budget_cache_dir", default=None,
        help="Take each frame's keep count from this mask cache, which should be the "
             "selector's own output. Without it --retention is used, which matches the "
             "rate on average but not frame by frame.")
    p.add_argument("--retention", type=float, default=0.06)
    p.add_argument("--block_px", type=int, default=32,
                   help="Tile side for --mode blocks.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rgb_dir", default=None)
    p.add_argument("--mask_cache_dir", default=None)
    add_dataset_args(p)
    return resolve(p.parse_args(), p)


def plane(rgb_path, rgb_dir, cache, h, w):
    seg = list(load_or_build_masks(rgb_path, rgb_dir, Path(cache), None,
                                   np.empty((h, w, 3), np.uint8), 0.0))
    return np.any(np.stack(seg), axis=0) if seg else np.zeros((h, w), bool)


def draw_pixels(contour, keep, rng):
    ys, xs = np.nonzero(contour)
    if keep >= len(ys):
        return contour.copy()
    idx = rng.choice(len(ys), size=keep, replace=False)
    out = np.zeros_like(contour)
    out[ys[idx], xs[idx]] = True
    return out


def draw_blocks(contour, keep, rng, block_px):
    """Accept whole tiles in random order until the budget is reached.

    The last accepted tile is taken in full rather than trimmed: trimming it would
    reintroduce a pixel-level choice, and the overshoot is one tile's worth of contour.
    """
    h, w = contour.shape
    gy, gx = (h + block_px - 1) // block_px, (w + block_px - 1) // block_px
    counts = np.zeros((gy, gx), np.int64)
    for i in range(gy):
        for j in range(gx):
            counts[i, j] = contour[i * block_px:(i + 1) * block_px,
                                   j * block_px:(j + 1) * block_px].sum()
    tiles = [(i, j) for i in range(gy) for j in range(gx) if counts[i, j]]
    rng.shuffle(tiles)
    out = np.zeros_like(contour)
    total = 0
    for i, j in tiles:
        if total >= keep:
            break
        sl = (slice(i * block_px, (i + 1) * block_px),
              slice(j * block_px, (j + 1) * block_px))
        out[sl] = contour[sl]
        total += counts[i, j]
    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rgb_dir, out_dir = Path(args.rgb_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_frames(args)

    kept, budgets = [], []
    for rgb_path, depth_path in tqdm(pairs, desc=f"random/{args.mode}"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(args.dataset)
        valid = valid_mask(args.dataset, gt)
        if args.dataset == "nyuv2":
            valid = valid & eigen_valid_mask(*gt.shape)
        h, w = gt.shape
        contour = plane(rgb_path, rgb_dir, args.mask_cache_dir, h, w) & valid
        n = int(contour.sum())
        if not n:
            continue
        if args.budget_cache_dir:
            keep = int((plane(rgb_path, rgb_dir, args.budget_cache_dir, h, w)
                        & valid).sum())
        else:
            keep = int(round(args.retention * n))
        keep = max(1, min(keep, n))
        sel = (draw_pixels(contour, keep, rng) if args.mode == "pixels"
               else draw_blocks(contour, keep, rng, args.block_px))
        kept.append(sel.sum() / n)
        budgets.append(keep / n)

        path = cache_path(rgb_path, rgb_dir, out_dir, "_seg.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = sel[None]
        np.savez_compressed(
            path, packed=np.packbits(arr.reshape(1, -1), axis=-1), n=1)

    summary = {"mode": args.mode, "seed": args.seed, "frames": len(kept),
               "retention": float(np.mean(kept)),
               "budget": float(np.mean(budgets)),
               "block_px": args.block_px if args.mode == "blocks" else None}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"残存率 {np.mean(kept) * 100:.2f}%  (予算 {np.mean(budgets) * 100:.2f}%)  "
          f"{len(kept)} 枚 -> {out_dir}")


if __name__ == "__main__":
    main()
