#!/usr/bin/env python
"""Sharpen Lotus's depth at YOLO-seg contours, with no ground truth and no training.

The three oracle curves separated what this operation needs from what it does not:

  precision   irrelevant. Adding twice as much spurious contour as there is real contour
              costs 0.8 points of 314. A contour in a flat region does nothing, because
              refilling each side with its own local value changes nothing where the two
              sides agree - the operation sharpens an existing step, it cannot invent one.
  recall      roughly linear. 5% coverage still returns +12.5%.
  position    the binding constraint. Gains survive about 1.7px of displacement.

Object masks land differently on each: 42.9% of their contour pixels mark no depth step
(harmless), they cover 7.9% of true discontinuities (the limit), and where a step does
exist they sit at a median 2.00px - right at the edge of what position tolerates.

Those were measured separately under a synthetic error model. Here the real contours run
through the real operation, so all three act at once and in their true proportions.

Variants:
  yolo          every mask contour pixel as a barrier
  yolo_ctrl     the same contours relocated - the control every measurement in this
                investigation has needed, since substituting anything into a region of
                that size tends to help on its own
  yolo_selected only the contour pixels where Lotus's OWN depth already shows a step.
                Precision being free predicts this is pointless; it is included because
                the filter also removes contour that is merely near a step rather than on
                it, and displaced contour is the one kind that hurts.
  perfect       GT discontinuities, as the ceiling this is measured against
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from eval_object_oracle_ceiling import _cache_path, load_or_build_masks, score
from eval_perfect_contour_ceiling import propagate_labels, refill_from_own_side
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from utils.align_space import add_align_space_arg, get_aligner
from utils.eval_frames import (add_dataset_args, depth_scale, list_frames,
                               valid_mask)

VARIANTS = ["baseline", "yolo", "yolo_ctrl", "yolo_selected", "perfect"]


def parse_args():
    p = argparse.ArgumentParser(description="Contour sharpening from real YOLO masks.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/yolo_seg")
    p.add_argument("--output_dir", type=str, default="output/eval_yolo_contour_sharpening")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--band_px", type=int, default=2)
    p.add_argument("--fill_radius", type=int, default=3,
                   help="Swept: 3 is the smallest radius reaching past the band, and the "
                        "most local same-side value gives the sharpest step.")
    p.add_argument("--select_t", type=float, default=5.0,
                   help="Threshold for 'Lotus already shows a step here', in percent.")
    p.add_argument("--select_px", type=int, default=3)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument(
        "--require_masks_in",
        type=str,
        default=None,
        help=(
            "Skip frames that have no mask in this other cache. Object masks exist for 521 "
            "of 654 NYUv2 frames while whole-scene contours exist for all of them, so "
            "comparing the two without this compares different populations - their "
            "baselines already differ, 0.0726 against 0.0693."
        ),
    )
    p.add_argument(
        "--masks_are_contours",
        action="store_true",
        help=(
            "The cache already holds contour planes rather than region masks "
            "(build_sam_masks.py --mode automatic). Taking contour_of() of a contour "
            "would return the outline of the line - two parallel strokes instead of one."
        ),
    )
    p.add_argument("--n_controls", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument(
        "--allow_missing_pred", action="store_true",
        help="Skip frames whose prediction is not cached instead of failing. Off by "
             "default: a partial cache scores a subset of the split without saying so, "
             "and the resulting number gets compared against a full run.")
    add_align_space_arg(p)
    add_dataset_args(p)
    return p.parse_args()


def contour_of(mask):
    k = np.ones((3, 3), np.uint8)
    u = mask.astype(np.uint8)
    return cv2.dilate(u, k).astype(bool) & ~cv2.erode(u, k).astype(bool)


def sharpen(base, contour, valid, ker, band_px, fill_radius):
    band = cv2.dilate(contour.astype(np.uint8), ker).astype(bool) & valid
    if not band.any():
        return base
    labels, _ = ndi.label(valid & ~band)
    labels = propagate_labels(labels, band, max_iter=band_px + 2)
    return refill_from_own_side(base, labels, band, fill_radius)


def main():
    args = parse_args()
    align = get_aligner(args.align_space)
    rng = np.random.default_rng(args.seed)
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_frames(args)
    if args.max_images:
        pairs = pairs[: args.max_images]

    th = np.linspace(5.0, 25.0, 11)
    wt = th / th.sum()
    ker = np.ones((2 * args.band_px + 1, 2 * args.band_px + 1), np.uint8)
    sel_k = np.ones((2 * args.select_px + 1, 2 * args.select_px + 1), np.uint8)
    bf1 = {v: [] for v in VARIANTS}
    ar = {v: [] for v in VARIANTS}
    d1 = {v: [] for v in VARIANTS}
    kept = []
    skips = Counter()

    def weighted(c):
        ok = np.isfinite(c)
        return float((c[ok] * wt[ok]).sum() / wt[ok].sum()) if ok.any() else np.nan

    for rgb_path, depth_path in tqdm(pairs, desc="yolo_sharpen"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(args.dataset)
        h, w = gt.shape
        valid = valid_mask(args.dataset, gt)
        if valid.sum() < 100:
            skips["too_few_valid_px"] += 1
            continue
        pp = _cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        if not pp.is_file():
            # A partial prediction cache is always a setup error, and skipping it
            # silently scores a subset of the split: two runs then report different
            # frame counts and get compared anyway.
            if not args.allow_missing_pred:
                raise SystemExit("\n".join([
                    f"no cached prediction for {rgb_path}",
                    f"  expected {pp}",
                    f"  from --pred_cache_dir {args.pred_cache_dir} "
                    f"+ res{args.processing_res}",
                    "Run inference for this split, or pass --allow_missing_pred to "
                    "skip the missing frames on purpose.",
                ]))
            skips["pred_missing"] += 1
            continue
        base = align(np.load(pp).astype(np.float64), gt, valid)
        if base is None:
            skips["alignment_failed"] += 1
            continue
        if args.require_masks_in:
            other = list(load_or_build_masks(
                rgb_path, rgb_dir, Path(args.require_masks_in), None,
                np.empty((h, w, 3), np.uint8), args.detection_score_thr))
            if not other or not np.any(np.stack(other)):
                skips["no_required_mask"] += 1
                continue
        seg = list(load_or_build_masks(rgb_path, rgb_dir, Path(args.mask_cache_dir), None,
                                       np.empty((h, w, 3), np.uint8), args.detection_score_thr))
        if not seg:
            skips["no_mask"] += 1
            continue
        plane = np.any(np.stack(seg), axis=0)
        cont = (plane if args.masks_are_contours else contour_of(plane)) & valid
        if not cont.any():
            skips["empty_contour"] += 1
            continue

        # keep contour pixels near a step the model itself already predicts
        near_pred = cv2.dilate(
            discontinuities(base, valid, args.select_t).astype(np.uint8), sel_k
        ).astype(bool)
        selected = cont & near_pred
        kept.append(float(selected.sum() / max(cont.sum(), 1)))

        variants = {
            "baseline": base,
            "yolo": sharpen(base, cont, valid, ker, args.band_px, args.fill_radius),
            "yolo_selected": sharpen(base, selected, valid, ker, args.band_px, args.fill_radius),
            "perfect": sharpen(base, discontinuities(gt, valid, args.t), valid, ker,
                               args.band_px, args.fill_radius),
        }
        acc = np.zeros_like(base)
        for _ in range(args.n_controls):
            dy, dx = int(rng.integers(h // 5, 4 * h // 5)), int(rng.integers(w // 5, 4 * w // 5))
            acc += sharpen(base, np.roll(cont, (dy, dx), axis=(0, 1)) & valid, valid,
                           ker, args.band_px, args.fill_radius)
        variants["yolo_ctrl"] = acc / args.n_controls

        for name, d in variants.items():
            bf1[name].append(weighted(boundary_f1(d, gt, valid, th)))
            _ar, _d1 = score(d, gt, valid)
            ar[name].append(_ar)
            d1[name].append(_d1)

    n = len(bf1["baseline"])
    summary = {"n_images": n, "n_frames_listed": len(pairs),
               "skipped": dict(sorted(skips.items())),
               "fill_radius": args.fill_radius, "band_px": args.band_px,
               "selected_frac_of_contour": float(np.mean(kept)) if kept else 0.0,
               "variants": {}}
    for v in VARIANTS:
        summary["variants"][v] = {"bf1": float(np.nanmean(bf1[v])),
                                  "abs_rel": float(np.mean(ar[v])),
                        "delta1": float(np.mean(d1[v]))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    V = summary["variants"]
    b = V["baseline"]
    print(f"\nYOLO contour sharpening  n={n}  radius={args.fill_radius}  "
          f"band=+-{args.band_px}px")
    print(f"\n{'variant':<16}{'BF1':>9}{'vs base':>10}{'abs_rel':>10}{'vs base':>10}")
    print("-" * 55)
    for v in VARIANTS:
        d = V[v]
        print(f"{v:<16}{d['bf1']:>9.4f}{100*(d['bf1']-b['bf1'])/b['bf1']:>9.1f}%"
              f"{d['abs_rel']:>10.5f}{100*(b['abs_rel']-d['abs_rel'])/b['abs_rel']:>9.1f}%")
    net = V["yolo"]["bf1"] - V["yolo_ctrl"]["bf1"]
    print(f"\nNET of control: {net:+.4f} BF1 ({100*net/b['bf1']:+.1f}% of baseline)")
    print(f"of the perfect-contour ceiling ({100*(V['perfect']['bf1']-b['bf1'])/b['bf1']:.1f}%): "
          f"{100*net/max(V['perfect']['bf1']-b['bf1'], 1e-9):.1f}% recovered")
    print(f"\nselection kept {summary['selected_frac_of_contour']*100:.1f}% of contour pixels")
    print(f"Saved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
