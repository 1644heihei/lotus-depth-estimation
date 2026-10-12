r"""Edge-preserving filters as a training-free alternative to the learned selector.

The obvious question about this work is why a classical edge-preserving filter would not
do the same job: both take a blurred depth map and an RGB image and try to put the depth
discontinuity back where the image says it is. The paper's controls were all built from
its own machinery, so this adds the two filters a reviewer would reach for first.

  guided     He et al.'s guided filter with the grayscale image as guide. Implemented
             here with box filters rather than taken from opencv-contrib, which is not
             installed and would be one more thing to match versions on.
  joint      A joint bilateral filter, the other standard choice: a Gaussian in space
             times a Gaussian in guide intensity.

Both are swept over their radius, since a filter that is too small cannot move a
boundary and one that is too large destroys the surfaces either side. Scored with the
same boundary F1, the same thresholds and the same alignment as everything else, so the
numbers go straight into the controls table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from eval_boundary_f1 import boundary_f1
from eval_regressor_predepth_nyuv2 import eigen_valid_mask
from utils.align_space import add_align_space_arg, get_aligner
from utils.dataset_paths import resolve
from utils.eval_frames import add_dataset_args, depth_scale, list_frames, valid_mask
from contour_selector.paths import cache_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--radii", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--eps", type=float, default=1e-4,
                   help="Guided filter regularisation, on a guide scaled to [0,1].")
    p.add_argument("--sigma_colour", type=float, default=0.1,
                   help="Joint bilateral range sigma, same scaling.")
    p.add_argument("--output_dir", default="output/eval_classical_filters")
    p.add_argument("--rgb_dir", default=None)
    p.add_argument("--pred_cache_dir", default=None)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--max_images", type=int, default=0)
    add_align_space_arg(p)
    add_dataset_args(p)
    return resolve(p.parse_args(), p)


def guided(depth, guide, radius, eps):
    """He et al., with a single-channel guide. Box filters, so O(1) in the radius."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean = lambda a: cv2.boxFilter(a, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mg, md = mean(guide), mean(depth)
    cov = mean(guide * depth) - mg * md
    var = mean(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = md - a * mg
    return mean(a) * guide + mean(b)


def joint_bilateral(depth, guide, radius, sigma_colour):
    """Gaussian in space times Gaussian in guide intensity, computed directly.

    cv2.ximgproc has this, but opencv-contrib is not installed and pinning it for one
    baseline is worse than twenty lines that can be read.
    """
    h, w = depth.shape
    sigma_space = max(radius / 2.0, 1e-3)
    out = np.zeros_like(depth)
    wsum = np.zeros_like(depth)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sw = np.exp(-(dy * dy + dx * dx) / (2 * sigma_space ** 2))
            gs = np.roll(np.roll(guide, dy, axis=0), dx, axis=1)
            ds = np.roll(np.roll(depth, dy, axis=0), dx, axis=1)
            rw = np.exp(-((gs - guide) ** 2) / (2 * sigma_colour ** 2))
            wgt = sw * rw
            out += wgt * ds
            wsum += wgt
    return out / np.maximum(wsum, 1e-9)


def main():
    args = parse_args()
    align = get_aligner(args.align_space)
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    pairs = list_frames(args)
    if args.max_images:
        pairs = pairs[: args.max_images]

    th = np.linspace(5.0, 25.0, 11)
    wt = th / th.sum()
    variants = ["baseline"] + [f"{m}_r{r}" for m in ("guided", "joint")
                               for r in args.radii]
    bf1 = {v: [] for v in variants}

    def weighted(c):
        ok = np.isfinite(c)
        return float((c[ok] * wt[ok]).sum() / wt[ok].sum()) if ok.any() else np.nan

    for rgb_path, depth_path in tqdm(pairs, desc="filters"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(args.dataset)
        v = valid_mask(args.dataset, gt)
        if args.dataset == "nyuv2":
            v = v & eigen_valid_mask(*gt.shape)
        pp = cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        if not pp.is_file():
            continue
        base = align(np.load(pp).astype(np.float64), gt, v)

        rgb = np.array(Image.open(rgb_path).convert("L")).astype(np.float64) / 255.0
        if rgb.shape != base.shape:
            rgb = cv2.resize(rgb, (base.shape[1], base.shape[0]),
                             interpolation=cv2.INTER_AREA)

        out = {"baseline": base}
        for r in args.radii:
            out[f"guided_r{r}"] = guided(base, rgb, r, args.eps)
            out[f"joint_r{r}"] = joint_bilateral(base, rgb, r, args.sigma_colour)
        for name, d in out.items():
            bf1[name].append(weighted(boundary_f1(d, gt, v, th)))

    base_mean = float(np.nanmean(bf1["baseline"]))
    rows = {k: float(np.nanmean(x)) for k, x in bf1.items()}
    summary = {"frames": len(bf1["baseline"]), "bf1": rows,
               "relative_pct": {k: (v / base_mean - 1) * 100 for k, v in rows.items()},
               "eps": args.eps, "sigma_colour": args.sigma_colour, "radii": args.radii}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{len(bf1['baseline'])} frames\n")
    print(f"{'variant':16}{'BF1':>10}{'vs baseline':>13}")
    for k, v in rows.items():
        print(f"{k:16}{v:>10.5f}{(v / base_mean - 1) * 100:>12.2f}%")


if __name__ == "__main__":
    main()
