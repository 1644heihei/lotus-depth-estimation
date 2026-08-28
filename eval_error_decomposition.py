#!/usr/bin/env python
"""Where does Lotus's depth error live, and where did the 512->768 gain land?

Phase 0-3 of docs/lotus_improvement_plan.md. The oracle measurement already showed
error is spread ~uniformly by area across object vs background. This cuts the image
a different way - by distance to a depth discontinuity - to answer the question that
decides whether tiled inference is worth building:

  Does Lotus's error concentrate at depth boundaries (where more latent capacity
  sharpens edges), or in smooth interiors (where it would not help)?

Restoring the official processing_res (512 -> 768) is a natural experiment: the same
scenes predicted at two latent capacities, one clearly better. Decomposing that gain
spatially reveals the mechanism. If the gain lands on boundaries, tiling - which buys
more latent capacity per source pixel - should extend it. If it lands on smooth
regions, the problem is global structure and tiling will not help.

Bands are computed from the GT depth map, so they are identical for both resolutions
and the comparison is like-for-like.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from evaluation.util.metric import abs_relative_difference
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs


def parse_args():
    p = argparse.ArgumentParser(description="Spatial error decomposition for Lotus on NYUv2.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument(
        "--pred_cache_dir",
        type=str,
        default="D:/lotus/data/oracle_cache/lotus_pred",
        help="Root of the resolution-scoped Lotus prediction cache built by eval_object_oracle_ceiling.py.",
    )
    p.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[512, 768],
        help="Resolutions to decompose and compare (must already be cached).",
    )
    p.add_argument("--output_dir", type=str, default="output/eval_error_decomposition")
    p.add_argument(
        "--boundary_thresh",
        type=float,
        default=0.10,
        help="Relative depth step (|grad|/depth) above which a GT pixel counts as a discontinuity.",
    )
    p.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20],
        help="Distance-to-boundary band edges in pixels; a final 'interior' band covers the rest.",
    )
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def gt_boundary_mask(gt: np.ndarray, valid: np.ndarray, thresh: float) -> np.ndarray:
    """Pixels sitting on a relative depth step larger than `thresh`.

    Relative rather than absolute, because a 10cm step matters at 1m and not at 8m -
    the metric being decomposed (abs_rel) is relative too.
    """
    d = np.where(valid, gt, np.nan)
    gy = np.abs(np.gradient(d, axis=0))
    gx = np.abs(np.gradient(d, axis=1))
    grad = np.fmax(np.nan_to_num(gy, nan=0.0), np.nan_to_num(gx, nan=0.0))
    rel = grad / np.maximum(gt, 1e-3)
    return (rel > thresh) & valid


def distance_bands(boundary: np.ndarray, valid: np.ndarray, edges: list[int]):
    """Yield (label, mask) for each distance-to-boundary band, plus the interior."""
    try:
        import cv2

        dist = cv2.distanceTransform(
            (~boundary).astype(np.uint8), cv2.DIST_L2, 3
        ).astype(np.float32)
    except Exception:  # pragma: no cover - cv2 is a hard dep elsewhere in this repo
        from scipy import ndimage

        dist = ndimage.distance_transform_edt(~boundary).astype(np.float32)

    prev = -1.0
    for e in edges:
        m = valid & (dist > prev) & (dist <= e)
        yield f"<={e}px", m
        prev = float(e)
    yield f">{edges[-1]}px (interior)", valid & (dist > edges[-1])


def aligned_depth(pred_disp: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    gt_disp, gt_nn = depth2disparity(depth=gt, return_mask=True)
    valid_nn = valid & gt_nn & (pred_disp > 0)
    if valid_nn.sum() < 100:
        return None
    disp_aligned, _, _ = align_depth_least_square(
        gt_arr=gt_disp,
        pred_arr=pred_disp,
        valid_mask_arr=valid_nn,
        return_scale_shift=True,
    )
    return np.clip(disparity2depth(np.clip(disp_aligned, 1e-3, None)), 1e-3, 10.0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    band_labels = [f"<={e}px" for e in args.bands] + [f">{args.bands[-1]}px (interior)"]
    # Per resolution: summed absolute relative error and pixel count per band.
    # Summing error*count (not averaging per-image means) keeps bands comparable when
    # a band is tiny in one image and large in another.
    acc = {r: {b: {"err": 0.0, "n": 0} for b in band_labels} for r in args.resolutions}
    rows = []

    for rgb_path, depth_path in tqdm(pairs, desc="error_decomposition"):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape[:2]
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        if valid.sum() < 100:
            continue

        boundary = gt_boundary_mask(gt, valid, args.boundary_thresh)
        bands = list(distance_bands(boundary, valid, args.bands))

        rel = rgb_path.relative_to(rgb_dir)
        depths = {}
        for r in args.resolutions:
            cache = Path(args.pred_cache_dir) / f"res{r}" / rel.parent / f"{rel.stem}_pred.npy"
            if not cache.is_file():
                raise FileNotFoundError(
                    f"Missing cached prediction: {cache}\n"
                    f"Build it first: python eval_object_oracle_ceiling.py --processing_res {r} ..."
                )
            pred = np.load(cache).astype(np.float64)
            if pred.shape != gt.shape:
                pred = np.array(
                    Image.fromarray(pred.astype(np.float32)).resize((w, h), Image.BILINEAR),
                    dtype=np.float64,
                )
            d = aligned_depth(pred, gt, valid)
            if d is None:
                break
            depths[r] = d
        if len(depths) != len(args.resolutions):
            continue

        row = {"filename": str(rel).replace("\\", "/"), "boundary_frac": float(boundary[valid].mean())}
        for r, d in depths.items():
            per_px = np.abs(d - gt) / np.maximum(gt, 1e-3)
            for label, m in bands:
                n = int(m.sum())
                if n == 0:
                    continue
                acc[r][label]["err"] += float(per_px[m].sum())
                acc[r][label]["n"] += n
            row[f"abs_rel_res{r}"] = float(
                abs_relative_difference(
                    torch.from_numpy(d), torch.from_numpy(gt), torch.from_numpy(valid)
                )
            )
        rows.append(row)

    if not rows:
        raise RuntimeError("No images scored.")

    total_px = {r: sum(v["n"] for v in acc[r].values()) for r in args.resolutions}
    total_err = {r: sum(v["err"] for v in acc[r].values()) for r in args.resolutions}

    summary = {"num_images": len(rows), "boundary_thresh": args.boundary_thresh, "bands": {}}
    for label in band_labels:
        entry = {}
        for r in args.resolutions:
            a = acc[r][label]
            entry[f"res{r}"] = {
                "abs_rel": a["err"] / max(a["n"], 1),
                "pixel_share": a["n"] / max(total_px[r], 1),
                "error_share": a["err"] / max(total_err[r], 1e-12),
            }
        if len(args.resolutions) == 2:
            lo, hi = args.resolutions
            drop = acc[lo][label]["err"] - acc[hi][label]["err"]
            entry["error_drop"] = drop
            entry["share_of_total_gain"] = drop / max(total_err[lo] - total_err[hi], 1e-12)
        summary["bands"][label] = entry

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lo, hi = (args.resolutions + [None, None])[:2]
    print(f"\nboundary = GT relative depth step > {args.boundary_thresh}")
    print(f"images: {len(rows)}\n")
    head = f"{'band':<22}{'px share':>10}"
    for r in args.resolutions:
        head += f"{'abs_rel@' + str(r):>13}"
    if hi:
        head += f"{'err share@' + str(hi):>14}{'of total gain':>15}"
    print(head)
    print("-" * len(head))
    for label in band_labels:
        e = summary["bands"][label]
        line = f"{label:<22}{e[f'res{args.resolutions[0]}']['pixel_share']*100:>9.1f}%"
        for r in args.resolutions:
            line += f"{e[f'res{r}']['abs_rel']:>13.5f}"
        if hi:
            line += f"{e[f'res{hi}']['error_share']*100:>13.1f}%{e['share_of_total_gain']*100:>14.1f}%"
        print(line)
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
