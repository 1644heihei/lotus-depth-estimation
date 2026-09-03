#!/usr/bin/env python
"""Can full-resolution RGB edges fix Lotus's ~3px boundary displacement?

docs/object_boundary_closure.md closed the object-detection route: YOLO contours sit a
median 14px from the true depth discontinuity while Lotus's own sit at 3px, so the
detector is strictly worse than the model it was meant to help. What that left is a
displacement, not a blur - Lotus puts its depth edges in roughly the right place, about
three pixels off.

Correcting three pixels needs something localised to about one. The RGB image at full
resolution is the one candidate that is not circular: Lotus only ever sees it resized to
768 and then compressed 8x by the VAE, so pixel-accurate image edges are genuinely
outside its output.

Two things have to hold for a snap-to-image-edge method to exist, and they are measured
separately because either alone is misleading:

  ALIGNMENT   true depth discontinuities must lie on RGB edges. If GT itself sits 4px
              from the nearest image edge, no image-based method can localise better.
  SPECIFICITY most RGB edges are texture and paint, not geometry. If only a small
              fraction of edge pixels mark a depth step, "nearest edge" pulls boundaries
              toward the wrong ones and snapping makes things worse.

Then the operation itself is simulated directly, with no method built: move each Lotus
discontinuity pixel to its nearest RGB edge and ask whether that lands it closer to a
true discontinuity than where it started.
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

from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path, align_to_gt
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs


def parse_args():
    p = argparse.ArgumentParser(description="RGB edge alignment with depth discontinuities.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--output_dir", type=str, default="output/eval_rgb_edge_alignment")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--canny", type=int, nargs=2, action="append",
                   help="Canny low/high pair; repeatable. Default sweeps three settings.")
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def stats(chunks):
    a = np.concatenate(chunks)
    return {
        "n_px": int(a.size),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "within_1px": float((a <= 1.0).mean()),
        "within_2px": float((a <= 2.0).mean()),
        "within_4px": float((a <= 4.0).mean()),
    }


def main():
    args = parse_args()
    settings = args.canny or [(50, 150), (100, 200), (150, 300)]
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    acc = {tuple(s): {"gt_to_edge": [], "lotus_to_edge": [], "edge_to_gt": [],
                      "before": [], "after": [], "edge_frac": []} for s in settings}
    base_lotus_to_gt = []

    for rgb_path, depth_path in tqdm(pairs, desc="rgb_edges"):
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

        gt_d = discontinuities(gt, valid, args.t)
        lo_d = discontinuities(base, valid, args.t)
        if not gt_d.any() or not lo_d.any():
            continue
        d_to_gt = ndi.distance_transform_edt(~gt_d)
        base_lotus_to_gt.append(d_to_gt[lo_d])

        grey = np.array(Image.open(rgb_path).convert("L"))
        for s in settings:
            lo_t, hi_t = s
            edge = cv2.Canny(grey, lo_t, hi_t).astype(bool) & valid
            e = acc[tuple(s)]
            if not edge.any():
                continue
            d_to_edge, (ey, ex) = ndi.distance_transform_edt(~edge, return_indices=True)
            e["gt_to_edge"].append(d_to_edge[gt_d])
            e["lotus_to_edge"].append(d_to_edge[lo_d])
            e["edge_to_gt"].append(d_to_gt[edge])
            e["edge_frac"].append(float(edge.sum() / max(valid.sum(), 1)))
            # the snap itself: move each Lotus discontinuity to its nearest RGB edge
            e["before"].append(d_to_gt[lo_d])
            e["after"].append(d_to_gt[ey[lo_d], ex[lo_d]])

    summary = {"t": args.t, "n_images": len(base_lotus_to_gt),
               "lotus_to_gt_baseline": stats(base_lotus_to_gt), "canny": {}}
    for s in settings:
        e = acc[tuple(s)]
        if not e["before"]:
            continue
        before = np.concatenate(e["before"])
        after = np.concatenate(e["after"])
        summary["canny"][f"{s[0]}_{s[1]}"] = {
            "edge_pixel_fraction": float(np.mean(e["edge_frac"])),
            "gt_disc_to_edge": stats(e["gt_to_edge"]),
            "lotus_disc_to_edge": stats(e["lotus_to_edge"]),
            "edge_to_gt_disc": stats(e["edge_to_gt"]),
            "snap": {
                "median_before": float(np.median(before)),
                "median_after": float(np.median(after)),
                "mean_before": float(before.mean()),
                "mean_after": float(after.mean()),
                "frac_improved": float((after < before).mean()),
                "frac_worsened": float((after > before).mean()),
            },
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    b = summary["lotus_to_gt_baseline"]
    print(f"\nRGB edge alignment  n={summary['n_images']} images  t={args.t}%")
    print(f"\nLotus discontinuity -> GT discontinuity (current state): "
          f"median {b['median']:.2f}px  <=2px {b['within_2px']*100:.1f}%")

    print(f"\n--- ALIGNMENT: do depth discontinuities lie on RGB edges? ---")
    print(f"{'Canny':<12}{'edge px':>9}{'GT->edge':>11}{'<=1px':>8}{'Lotus->edge':>13}{'<=1px':>8}")
    print("-" * 61)
    for k, v in summary["canny"].items():
        g, l = v["gt_disc_to_edge"], v["lotus_disc_to_edge"]
        print(f"{k:<12}{v['edge_pixel_fraction']*100:>8.1f}%{g['median']:>11.2f}"
              f"{g['within_1px']*100:>7.1f}%{l['median']:>13.2f}{l['within_1px']*100:>7.1f}%")

    print(f"\n--- SPECIFICITY: how many RGB edges are actually depth steps? ---")
    print(f"{'Canny':<12}{'edge->GT median':>17}{'<=1px':>8}{'<=2px':>8}")
    print("-" * 45)
    for k, v in summary["canny"].items():
        e = v["edge_to_gt_disc"]
        print(f"{k:<12}{e['median']:>17.2f}{e['within_1px']*100:>7.1f}%{e['within_2px']*100:>7.1f}%")

    print(f"\n--- THE SNAP: move each Lotus discontinuity to its nearest RGB edge ---")
    print(f"{'Canny':<12}{'median before':>15}{'median after':>14}{'improved':>11}{'worsened':>11}")
    print("-" * 63)
    for k, v in summary["canny"].items():
        s = v["snap"]
        print(f"{k:<12}{s['median_before']:>15.2f}{s['median_after']:>14.2f}"
              f"{s['frac_improved']*100:>10.1f}%{s['frac_worsened']*100:>10.1f}%")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
