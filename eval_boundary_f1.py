#!/usr/bin/env python
"""Boundary F1 for Lotus, and how much of the boundary gap an oracle could close.

The band decomposition (docs/object_band_decomposition.md) put the object-boundary oracle
at +0.8% to +2.6% abs_rel and I read that as "too small to chase". That reading used the
wrong instrument. abs_rel averages over pixels, and a +-2px band is 1.7% of them, so a
metric that weights every pixel equally cannot see a boundary method - which is exactly
why the field scores boundaries with F1 instead.

The gap is also bigger than I claimed. Depth Pro (arXiv 2410.02073) Table: boundary F1 on
Sintel is 0.409 for Depth Pro, 0.228 Depth Anything V2, 0.181 MiDaS, and 0.068 Marigold -
the latent-diffusion model is last by 6x, and Lotus inherits Marigold's VAE.

Metric follows Depth Pro exactly. Occluding contours come from the ratio between
neighbouring pixels,

    c_d(i,j) = [ d(j)/d(i) > 1 + t/100 ]

which is scale-invariant, so it applies to relative-depth predictions. Thresholds sweep
t = 5..25; F1 is symmetric in P and R, so the precision/recall labelling convention does
not affect the reported number.

Reported per variant:
  baseline     aligned Lotus
  oracle_band  the +-k px band around YOLO-seg silhouettes given GT depth
  oracle_mask  whole silhouettes given GT depth
  oracle_all   everything given GT - sanity check, must be 1.0

CAVEAT: NYUv2's labelled GT is inpainted, so its own occluding contours are softened.
Depth Pro scores this metric on synthetic and matting data for that reason. Treat the
absolute value here as indicative and the baseline-to-oracle gap as the finding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_object_oracle_ceiling import (
    _cache_path,
    align_to_gt,
    load_or_build_masks,
    oracle_replace,
    score,
)
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs

VARIANTS = ["baseline", "oracle_band", "oracle_band_ctrl", "oracle_mask", "oracle_all"]


def parse_args():
    p = argparse.ArgumentParser(description="Depth Pro boundary F1 for Lotus + oracles.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/yolo_seg")
    p.add_argument("--output_dir", type=str, default="output/eval_boundary_f1")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--band_px", type=int, default=8)
    p.add_argument("--t_min", type=float, default=5.0)
    p.add_argument("--t_max", type=float, default=25.0)
    p.add_argument("--n_thresholds", type=int, default=11)
    p.add_argument("--n_controls", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def contours(depth, valid, t):
    """Depth Pro's occluding contours: neighbour ratio above 1 + t/100, both directions.

    Returns the contour indicator and the pair-validity mask for each of 4 neighbour
    offsets, so precision and recall only ever count pairs where both pixels have GT.
    """
    r = 1.0 + t / 100.0
    out = []
    for axis, step in ((1, 1), (0, 1)):
        a = depth
        b = np.roll(depth, -step, axis=axis)
        va = valid & np.roll(valid, -step, axis=axis)
        # drop the wrapped-around edge row/column
        if axis == 1:
            va[:, -step:] = False
        else:
            va[-step:, :] = False
        out.append(((b / np.maximum(a, 1e-9) > r) & va, va))
        out.append(((a / np.maximum(b, 1e-9) > r) & va, va))
    return out


def boundary_f1(pred, gt, valid, thresholds):
    """Weighted-mean F1 over the threshold sweep, plus the per-threshold curve."""
    f1s = []
    for t in thresholds:
        cp = contours(pred, valid, t)
        cg = contours(gt, valid, t)
        tp = sp = sg = 0
        for (p_c, _), (g_c, _) in zip(cp, cg):
            tp += int((p_c & g_c).sum())
            sp += int(p_c.sum())
            sg += int(g_c.sum())
        if sp == 0 or sg == 0:
            f1s.append(0.0 if (sp or sg) else np.nan)
            continue
        prec, rec = tp / sp, tp / sg
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return np.array(f1s, dtype=float)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    mask_cache = Path(args.mask_cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.linspace(args.t_min, args.t_max, args.n_thresholds)
    # Depth Pro weights the sweep "towards high threshold values"
    weights = thresholds / thresholds.sum()

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]
    logging.info("Images: %d  thresholds: %s", len(pairs), np.round(thresholds, 1).tolist())

    rng = np.random.default_rng(args.seed)
    k = 2 * args.band_px + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    curves = {v: [] for v in VARIANTS}
    absrel = {v: [] for v in VARIANTS}

    for rgb_path, depth_path in tqdm(pairs, desc="boundary_f1"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        if valid.sum() < 100:
            continue
        pp = _cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        if not pp.is_file():
            continue
        base = align_to_gt(np.load(pp).astype(np.float64), gt, valid)
        if base is None:
            continue

        seg = list(load_or_build_masks(rgb_path, rgb_dir, mask_cache, None,
                                       np.empty((h, w, 3), np.uint8), args.detection_score_thr))
        seg_u = np.any(np.stack(seg), axis=0) if seg else np.zeros((h, w), bool)
        if not seg_u.any():
            continue
        u8 = seg_u.astype(np.uint8)
        band = cv2.dilate(u8, ker).astype(bool) & ~cv2.erode(u8, ker).astype(bool)

        variants = {
            "baseline": base,
            "oracle_band": oracle_replace(base, gt, band, valid),
            "oracle_mask": oracle_replace(base, gt, seg_u, valid),
            # Control: the same band relocated. It hands GT to just as many pixels but
            # not to the object's contour, so a real boundary effect must beat it.
            "oracle_band_ctrl": None,
            "oracle_all": oracle_replace(base, gt, np.ones_like(valid), valid),
        }
        cf = np.zeros(len(thresholds))
        ca = 0.0
        for _ in range(args.n_controls):
            dy = int(rng.integers(h // 5, 4 * h // 5))
            dx = int(rng.integers(w // 5, 4 * w // 5))
            dc = oracle_replace(base, gt, np.roll(band, (dy, dx), axis=(0, 1)), valid)
            cf += boundary_f1(dc, gt, valid, thresholds)
            ca += score(dc, gt, valid)[0]
        curves["oracle_band_ctrl"].append(cf / args.n_controls)
        absrel["oracle_band_ctrl"].append(ca / args.n_controls)
        del variants["oracle_band_ctrl"]

        for name, d in variants.items():
            curves[name].append(boundary_f1(d, gt, valid, thresholds))
            absrel[name].append(score(d, gt, valid)[0])

    n = len(curves["baseline"])
    summary = {"n_images": n, "band_px": args.band_px,
               "thresholds": thresholds.tolist(), "variants": {}}
    for v in VARIANTS:
        c = np.nanmean(np.stack(curves[v]), axis=0)
        summary["variants"][v] = {
            "bf1": float((c * weights).sum()),
            "bf1_curve": c.tolist(),
            "abs_rel": float(np.mean(absrel[v])),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    V = summary["variants"]
    b = V["baseline"]
    print(f"\nBoundary F1 (Depth Pro definition)  n={n}  band=+-{args.band_px}px")
    print(f"\n{'variant':<14}{'BF1':>9}{'vs base':>10}{'abs_rel':>10}{'vs base':>10}")
    print("-" * 53)
    for v in VARIANTS:
        d = V[v]
        df1 = 100.0 * (d["bf1"] - b["bf1"]) / max(b["bf1"], 1e-9)
        dar = 100.0 * (b["abs_rel"] - d["abs_rel"]) / max(b["abs_rel"], 1e-9)
        print(f"{v:<14}{d['bf1']:>9.4f}{df1:>9.1f}%{d['abs_rel']:>10.5f}{dar:>9.1f}%")
    print("\nBF1 vs threshold t:")
    print(f"{'t':>6}" + "".join(f"{v[:9]:>11}" for v in VARIANTS))
    for i, t in enumerate(thresholds):
        print(f"{t:>6.1f}" + "".join(f"{V[v]['bf1_curve'][i]:>11.4f}" for v in VARIANTS))
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
