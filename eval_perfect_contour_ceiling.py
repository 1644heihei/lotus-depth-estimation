#!/usr/bin/env python
"""If you knew exactly where the depth discontinuities are, how much boundary is recoverable?

The band oracle in docs/boundary_f1_findings.md is worth +313% BF1 net of control, but it
hands over ground-truth DEPTH inside the band. A contour estimator would supply only the
LOCATION of the discontinuities, and the depth values would still be Lotus's own. Those are
different ceilings, and it is the second that decides whether estimating contours is worth
building - a question raised by the finding that YOLO contours, once you exclude the 42.9%
that mark no depth step, localise to a median 2.00px and are therefore not the problem.

The contour is used as a BARRIER, never as a source of values. Pixels in a thin band around
it are discarded and refilled from the same side, where "same side" is decided by
connectivity with the band removed - so nothing crosses the contour, and the step lands
exactly on it. Ground truth supplies the barrier's position and nothing else.

Reported against two references: Lotus untouched, and the band filled with GT depth. The
gap between the two oracles is the part of the boundary gain that needs correct depth
VALUES rather than correct discontinuity POSITIONS - and no contour estimator, however
good, can reach it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_boundary_f1 import boundary_f1
from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path, align_to_gt, oracle_replace, score
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs

VARIANTS = ["baseline", "contour_only", "contour_only_ctrl", "band_gt_depth"]


def parse_args():
    p = argparse.ArgumentParser(description="Ceiling of a perfect contour, without GT depth.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--output_dir", type=str, default="output/eval_perfect_contour")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--t", type=float, default=10.0, help="Discontinuity threshold, percent.")
    p.add_argument("--band_px", type=int, default=2, help="Half-width of the band refilled.")
    p.add_argument("--fill_radius", type=int, default=7)
    p.add_argument("--n_controls", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def propagate_labels(labels: np.ndarray, band: np.ndarray, max_iter: int) -> np.ndarray:
    """Grow the free regions' labels into the band one pixel at a time.

    A flood that advances a pixel per step cannot cross a band wider than the distance it
    has travelled, so the two sides meet inside the band instead of one leaking into the
    other. Ties at the meeting line go to the higher label id, which is arbitrary and
    harmless - either side is a legitimate answer for a pixel equidistant from both.
    """
    # float32 rather than int32: cv2.dilate has no int32 kernel, and label ids stay far
    # below 2^24 so the float carries them exactly
    out = labels.astype(np.float32)
    todo = band & (labels == 0)
    k = np.ones((3, 3), np.uint8)
    for _ in range(max_iter):
        if not todo.any():
            break
        grown = cv2.dilate(out, k)
        take = todo & (grown > 0)
        out[take] = grown[take]
        todo &= ~take
    return out.astype(np.int32)


def refill_from_own_side(depth, labels, band, radius):
    """Give each band pixel the local mean of its own side's depth.

    Local rather than global so the surface's own shape survives, restricted to one label
    so the mean never averages across the discontinuity - which is the whole point. Uses
    only the prediction's values; the contour contributed position alone.
    """
    out = depth.copy()
    d = 2 * radius + 1
    blur = lambda a: cv2.blur(a, (d, d), borderType=cv2.BORDER_REFLECT)
    for lab in np.unique(labels[band]):
        if lab == 0:
            continue
        src = ((labels == lab) & ~band).astype(np.float64)
        if not src.any():
            continue
        num, den = blur(depth * src), blur(src)
        target = band & (labels == lab) & (den > 1e-6)
        out[target] = num[target] / den[target]
    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    thresholds = np.linspace(5.0, 25.0, 11)
    weights = thresholds / thresholds.sum()
    ker = np.ones((2 * args.band_px + 1, 2 * args.band_px + 1), np.uint8)
    curves = {v: [] for v in VARIANTS}
    absrel = {v: [] for v in VARIANTS}

    for rgb_path, depth_path in tqdm(pairs, desc="perfect_contour"):
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

        disc = discontinuities(gt, valid, args.t)
        if not disc.any():
            continue
        band = cv2.dilate(disc.astype(np.uint8), ker).astype(bool) & valid
        free = valid & ~band
        labels, _ = ndi.label(free)
        labels = propagate_labels(labels, band, max_iter=args.band_px + 2)

        variants = {
            "baseline": base,
            "contour_only": refill_from_own_side(base, labels, band, args.fill_radius),
            "band_gt_depth": oracle_replace(base, gt, band, valid),
        }

        # Control: the same operation driven by a relocated contour. It refills just as
        # many pixels with just as local a mean, but at a place with no discontinuity, so
        # only the excess over this belongs to knowing where the contour is.
        acc = np.zeros_like(base)
        for _ in range(args.n_controls):
            dy, dx = int(rng.integers(h // 5, 4 * h // 5)), int(rng.integers(w // 5, 4 * w // 5))
            d2 = np.roll(disc, (dy, dx), axis=(0, 1))
            b2 = cv2.dilate(d2.astype(np.uint8), ker).astype(bool) & valid
            l2, _ = ndi.label(valid & ~b2)
            l2 = propagate_labels(l2, b2, max_iter=args.band_px + 2)
            acc += refill_from_own_side(base, l2, b2, args.fill_radius)
        variants["contour_only_ctrl"] = acc / args.n_controls

        for name, d in variants.items():
            c = boundary_f1(d, gt, valid, thresholds)
            ok = np.isfinite(c)
            curves[name].append(float((c[ok] * weights[ok]).sum() / weights[ok].sum())
                                if ok.any() else np.nan)
            absrel[name].append(score(d, gt, valid)[0])

    n = len(curves["baseline"])
    summary = {"n_images": n, "t": args.t, "band_px": args.band_px,
               "fill_radius": args.fill_radius, "variants": {}}
    for v in VARIANTS:
        summary["variants"][v] = {"bf1": float(np.nanmean(curves[v])),
                                  "abs_rel": float(np.mean(absrel[v]))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    V = summary["variants"]
    b = V["baseline"]
    print(f"\nPerfect-contour ceiling  n={n}  t={args.t}%  band=+-{args.band_px}px")
    print(f"\n{'variant':<20}{'BF1':>9}{'vs base':>10}{'abs_rel':>10}{'vs base':>10}")
    print("-" * 59)
    for v in VARIANTS:
        d = V[v]
        print(f"{v:<20}{d['bf1']:>9.4f}{100*(d['bf1']-b['bf1'])/b['bf1']:>9.1f}%"
              f"{d['abs_rel']:>10.5f}{100*(b['abs_rel']-d['abs_rel'])/b['abs_rel']:>9.1f}%")
    net = V["contour_only"]["bf1"] - V["contour_only_ctrl"]["bf1"]
    print(f"\ncontour position alone, net of control: {net:+.4f} BF1 "
          f"({100*net/b['bf1']:+.1f}% of baseline)")
    print(f"with GT depth in the band instead      : "
          f"{V['band_gt_depth']['bf1'] - b['bf1']:+.4f} BF1")
    print("\nThe difference is what needs correct depth VALUES, not correct positions -")
    print("no contour estimator can reach it.")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
