#!/usr/bin/env python
"""Figure panels showing what the sharpening actually does to a depth map.

The paper's numbers are a boundary metric: BF1 rises 0.0884 -> 0.0989 while AbsRel and
delta1 barely move, because those two average over every pixel and the image is mostly
flat surface. A reader who does not already know that reads "AbsRel unchanged" as "nothing
happened", so the change has to be visible somewhere. This draws it.

Each frame gets two rows. The top row is the whole image - RGB, the baseline prediction,
the sharpened result, ground truth - so the reader sees that nothing outside the contours
moved. The bottom row is a crop around where the sharpening changed the most, which is
where the step the model blurred becomes a step again.

Frames are ranked by ABSOLUTE BF1 gain, not relative, and only frames the depth model
already handles well are eligible. Ranking by relative gain picks the opposite: the top of
that list is frames whose baseline BF1 is near zero because the prediction itself is broken
(the worst frame in NYUv2 correlates 0.22 with GT against a median of 0.84), where any
change looks enormous in percent. A figure built from those would be showing the sharpening
rescuing a failure, not improving a good prediction. The frame's AbsRel is printed so a
reader can check that for themselves.

The last row is what BF1 actually compares: the depth discontinuities each map contains.
That is where a 2px band moving becomes visible at all - in the depth maps themselves the
change is 1.5% of pixels and invisible at page size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contour_selector import cache_path, load_candidates  # noqa: E402
from eval_boundary_f1 import boundary_f1  # noqa: E402
from eval_perfect_contour_ceiling import (propagate_labels,  # noqa: E402
                                          refill_from_own_side)
from eval_yolo_contour_sharpening import sharpen  # noqa: E402
from utils.align_space import add_align_space_arg, get_aligner  # noqa: E402
from utils.eval_frames import (add_dataset_args, depth_scale,  # noqa: E402
                               list_frames, valid_mask)

NYU = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar"


def parse_args():
    p = argparse.ArgumentParser(description="Qualitative panels for contour sharpening.")
    p.add_argument("--rgb_dir", type=str, default=f"{NYU}/test")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/marigold_v2_pred")
    p.add_argument("--processing_res", type=int, default=640)
    p.add_argument("--mask_cache_dir", type=str,
                   default="D:/lotus/data/oracle_cache/sam_auto48_sel_raw",
                   help="The contours to sharpen at - normally the selector's output.")
    p.add_argument("--sam_cache_dir", type=str, default="D:/lotus/data/oracle_cache/sam_auto48",
                   help="Every SAM contour, drawn as the candidates the selector chose from.")
    p.add_argument("--out_dir", type=str, default="output/figures/sharpening")
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--band_px", type=int, default=2)
    p.add_argument("--fill_radius", type=int, default=3)
    p.add_argument("--n_panels", type=int, default=6)
    p.add_argument("--crop", type=int, default=72,
                   help="Side of the zoom crop, in pixels. Small: the band is 2px wide, "
                        "so a crop big enough to show context hides the effect entirely.")
    p.add_argument("--max_abs_rel", type=float, default=0.06,
                   help="Only draw frames the depth model already handles this well, so "
                        "the figure shows sharpening a good prediction rather than "
                        "rescuing a broken one. 0.06 is the split's median.")
    p.add_argument("--cmap", type=str, default="magma")
    p.add_argument("--dpi", type=int, default=200)
    add_align_space_arg(p)
    add_dataset_args(p)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def frame_bf1(depth, gt, valid, thresholds, weights):
    """The split's own BF1 definition, for one frame."""
    c = np.asarray(boundary_f1(depth, gt, valid, thresholds), dtype=np.float64)
    ok = np.isfinite(c)
    if not ok.any():
        return float("nan")
    return float((c[ok] * weights[ok]).sum() / weights[ok].sum())


def zoom_box(changed, valid, side, shape):
    """A window centred on the densest change, clipped to the frame.

    Centred on where the sharpening moved the depth most, because that is the claim: a
    crop chosen anywhere else would be honest but uninformative.
    """
    h, w = shape
    side = min(side, h, w)
    weight = ndi.uniform_filter((changed & valid).astype(np.float32), size=side // 2)
    cy, cx = np.unravel_index(int(np.argmax(weight)), weight.shape)
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    return y0, x0, side


def colorise(depth, valid, lo, hi, cmap):
    """Depth to RGB on a shared scale, so the three maps are comparable by eye."""
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (plt.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 255
    return rgb


def edge_overlay(depth, gt, valid, t):
    """The discontinuities a depth map contains, drawn against the ones GT contains.

    This is what BF1 scores: green where the map agrees with GT, red where it invents a
    step, grey where GT has one the map misses.
    """
    from eval_mask_contour_localization import discontinuities
    d = discontinuities(depth, valid, t)
    g = discontinuities(gt, valid, t)
    img = np.full(depth.shape + (3,), 255, np.uint8)
    img[g & ~d] = (190, 190, 190)
    img[d & ~g] = (235, 70, 70)
    img[d & g] = (30, 150, 60)
    return img


def draw_panel(path, rgb, base, sharp, gt, valid, kept, cand, box, title, cmap, dpi, t):
    y0, x0, s = box
    lo, hi = np.percentile(gt[valid], [2, 98])
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))
    dep = {"Baseline": colorise(base, valid, lo, hi, cmap),
           "Sharpened": colorise(sharp, valid, lo, hi, cmap),
           "Ground truth": colorise(gt, valid, lo, hi, cmap)}

    overlay = rgb.copy()
    k2 = np.ones((2, 2), np.uint8)
    overlay[cv2.dilate(cand.astype(np.uint8), k2).astype(bool)] = (90, 160, 255)
    overlay[cv2.dilate(kept.astype(np.uint8), k2).astype(bool)] = (255, 60, 60)

    fig, ax = plt.subplots(3, 4, figsize=(13.2, 9.6))
    row0 = [("RGB", rgb), ("Baseline", dep["Baseline"]),
            ("Sharpened", dep["Sharpened"]), ("Ground truth", dep["Ground truth"])]
    for j, (name, im) in enumerate(row0):
        ax[0, j].imshow(im)
        ax[0, j].set_title(name, fontsize=11)
        ax[0, j].add_patch(plt.Rectangle((x0, y0), s, s, fill=False, color="#00d0ff", lw=1.6))
        ax[1, j].imshow(im[sl])
    row2 = [("SAM contours (blue) / kept (red)", overlay[sl]),
            ("Baseline edges", edge_overlay(base, gt, valid, t)[sl]),
            ("Sharpened edges", edge_overlay(sharp, gt, valid, t)[sl]),
            ("GT edges", edge_overlay(gt, gt, valid, t)[sl])]
    for j, (name, im) in enumerate(row2):
        ax[2, j].imshow(im)
        ax[2, j].set_title(name, fontsize=9)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    ax[1, 0].set_ylabel("zoom", fontsize=11)
    ax[2, 0].set_ylabel("edges (zoom)", fontsize=11)
    fig.text(0.5, 0.055, "edges: green = agrees with GT,  red = invented,  grey = missed",
             ha="center", fontsize=9, color="#444444")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    mask_dir, sam_dir = Path(args.mask_cache_dir), Path(args.sam_cache_dir)
    out_dir = Path(args.out_dir)
    align = get_aligner(args.align_space)

    th = np.linspace(5.0, 25.0, 11)
    wt = th / th.sum()
    ker = np.ones((2 * args.band_px + 1, 2 * args.band_px + 1), np.uint8)

    pairs = list_frames(args)
    if args.max_images:
        pairs = pairs[: args.max_images]

    scored = []
    for rgb_path, depth_path in tqdm(pairs, desc="score"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(args.dataset)
        valid = valid_mask(args.dataset, gt)
        if valid.sum() < 100:
            continue
        pp = cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        mp = cache_path(rgb_path, rgb_dir, mask_dir, "_seg.npz")
        if not (pp.is_file() and mp.is_file()):
            continue
        base = align(np.load(pp).astype(np.float64), gt, valid)
        if base is None:
            continue
        kept = load_candidates(rgb_path, rgb_dir, mask_dir, valid)
        if not kept.any():
            continue
        sharp = sharpen(base, kept, valid, ker, args.band_px, args.fill_radius)
        b0 = frame_bf1(base, gt, valid, th, wt)
        b1 = frame_bf1(sharp, gt, valid, th, wt)
        if not (np.isfinite(b0) and np.isfinite(b1)) or b0 <= 0:
            continue
        abs_rel = float(np.mean(np.abs(base[valid] - gt[valid]) / gt[valid]))
        scored.append({"rgb": str(rgb_path), "depth": str(depth_path),
                       "bf1_base": b0, "bf1_sharp": b1, "gain": b1 - b0,
                       "gain_pct": (b1 / b0 - 1) * 100, "abs_rel": abs_rel})

    n_all = len(scored)
    # Absolute gain, and only frames the model already handles: ranking by relative gain
    # puts the broken predictions first, since a near-zero baseline makes any change look
    # enormous in percent.
    eligible = sorted((r for r in scored if r["abs_rel"] <= args.max_abs_rel),
                      key=lambda r: -r["gain"])
    n = len(eligible)
    print(f"\n{n_all} frames scored.  "
          f"median gain {np.median([r['gain_pct'] for r in scored]):+.2f}%  "
          f"improved {sum(r['gain_pct'] > 0 for r in scored)}/{n_all}")
    print(f"{n} eligible (AbsRel <= {args.max_abs_rel}); ranked by ABSOLUTE BF1 gain")
    assert n, "no frame passed --max_abs_rel; raise it"

    # Panels from across the ranking, not only the top, so the figure is not a best case
    # presented as a typical one.
    picks = sorted({int(round(q * (n - 1))) for q in
                    np.linspace(0, 0.5, args.n_panels)})
    for rank in picks:
        r = eligible[rank]
        rgb_path, depth_path = Path(r["rgb"]), Path(r["depth"])
        gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(args.dataset)
        valid = valid_mask(args.dataset, gt)
        base = align(np.load(cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
                             ).astype(np.float64), gt, valid)
        kept = load_candidates(rgb_path, rgb_dir, mask_dir, valid)
        cand = load_candidates(rgb_path, rgb_dir, sam_dir, valid)
        sharp = sharpen(base, kept, valid, ker, args.band_px, args.fill_radius)
        changed = np.abs(sharp - base) > 1e-6
        box = zoom_box(changed, valid, args.crop, gt.shape)
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        title = (f"{rgb_path.parent.name}/{rgb_path.stem}    "
                 f"BF1 {r['bf1_base']:.4f} -> {r['bf1_sharp']:.4f} "
                 f"({r['gain_pct']:+.1f}%)    AbsRel {r['abs_rel']:.3f}    "
                 f"rank {rank + 1}/{n} by absolute gain    "
                 f"kept {kept.sum() / max(cand.sum(), 1) * 100:.1f}% of SAM contour")
        out = out_dir / f"rank{rank + 1:04d}_{rgb_path.parent.name}_{rgb_path.stem}.png"
        draw_panel(out, rgb, base, sharp, gt, valid, kept, cand, box, title,
                   args.cmap, args.dpi, args.t)
        print(f"  {out}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranking.json").write_text(json.dumps(scored, indent=1), encoding="utf-8")
    print(f"\nranking -> {out_dir / 'ranking.json'}")


if __name__ == "__main__":
    main()
