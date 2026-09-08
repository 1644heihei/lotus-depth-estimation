#!/usr/bin/env python
"""Is a mask contour in the wrong place, or in a place with no depth step to find?

docs/object_boundary_closure.md measured YOLO-seg contours at a median 14.1px from the
nearest true depth discontinuity and closed the direction on that. The number pools two
different things, and they have different fixes:

  SELECTION    a contour pixel where no depth step exists at all - a chair against a wall
               at the same distance, an object meeting the floor. The mask is not wrong;
               it marks a semantic boundary that is not a geometric one. Fixed by choosing
               which contour pixels to trust, which needs no better mask.
  LOCALISATION a contour pixel where a step does exist, but several pixels away. Fixed
               only by a sharper mask.

If selection dominates, the +313% BF1 headroom is reachable with the masks already in
hand. If localisation dominates, YOLO-seg's resolution is the wall.

Run on Hypersim as well as NYUv2, because NYUv2's labelled depth is inpainted and its own
discontinuities are smoothed - part of the 14.1px may be the ground truth's error rather
than the detector's. Hypersim is synthetic and exact.
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
from eval_object_oracle_ceiling import load_or_build_masks
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from utils.hypersim_holdout import load_split


def parse_args():
    p = argparse.ArgumentParser(description="Split contour error into selection and localisation.")
    p.add_argument("--dataset", choices=["nyuv2", "hypersim"], default="hypersim")
    p.add_argument(
        "--nyu_rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/yolo_seg")
    p.add_argument("--hypersim_root", type=str, default="D:/lotus/data/hypersim_processed/train")
    p.add_argument("--hypersim_masks", type=str, default="D:/lotus/data/hypersim_sem_masks/train")
    p.add_argument("--holdout_split", type=str, default="datasets/hypersim_holdout.json")
    p.add_argument("--output_dir", type=str, default="output/eval_contour_decomposition")
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument(
        "--window",
        type=float,
        default=8.0,
        help="A contour pixel counts as marking a real step when one lies within this far.",
    )
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument(
        "--masks_are_contours",
        action="store_true",
        help="The cache holds contour planes, not region masks (--mode automatic).",
    )
    p.add_argument("--max_images", type=int, default=300)
    return p.parse_args()


def contour_of(mask: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    u = mask.astype(np.uint8)
    return cv2.dilate(u, k).astype(bool) & ~cv2.erode(u, k).astype(bool)


def nyu_frames(args):
    rgb_dir = Path(args.nyu_rgb_dir)
    for rgb_path, depth_path in list_nyu_pairs(rgb_dir)[: args.max_images]:
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        seg = list(load_or_build_masks(rgb_path, rgb_dir, Path(args.mask_cache_dir), None,
                                       np.empty((h, w, 3), np.uint8), args.detection_score_thr))
        if not seg:
            continue
        yield gt, valid, np.any(np.stack(seg), axis=0)


def hypersim_frames(args):
    root = Path(args.hypersim_root)
    masks = Path(args.hypersim_masks)
    for rel in load_split(args.holdout_split)["eval_frames"][: args.max_images]:
        rel = Path(rel)
        depth = root / rel.parent / rel.name.replace("rgb_", "depth_plane_")
        mask_p = masks / rel
        if not depth.is_file() or not mask_p.is_file():
            continue
        # raw uint16: the discontinuity test is a depth RATIO, so the scale cancels
        gt = np.array(Image.open(depth)).astype(np.float64)
        m = np.array(Image.open(mask_p)) > 127
        if not m.any():
            continue
        yield gt, np.isfinite(gt) & (gt > 0), m


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = hypersim_frames(args) if args.dataset == "hypersim" else nyu_frames(args)

    near, far, rev, n = [], 0, [], 0
    total_contour = 0
    for gt, valid, seg in tqdm(frames, desc=f"{args.dataset}"):
        cont = (seg if args.masks_are_contours else contour_of(seg)) & valid
        disc = discontinuities(gt, valid, args.t)
        if not cont.any() or not disc.any():
            continue
        n += 1
        d = ndi.distance_transform_edt(~disc)[cont]
        total_contour += d.size
        near.append(d[d <= args.window])
        far += int((d > args.window).sum())
        rev.append(ndi.distance_transform_edt(~cont)[disc])

    d_all = np.concatenate([np.concatenate(near), np.full(far, np.inf)])
    d_near = np.concatenate(near)
    d_rev = np.concatenate(rev)

    def stats(a):
        a = a[np.isfinite(a)]
        return {
            "n_px": int(a.size), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            **{f"within_{k}px": float((a <= k).mean()) for k in (1, 2, 4)},
        }

    summary = {
        "dataset": args.dataset, "n_images": n, "t": args.t, "window": args.window,
        "contour_px": total_contour,
        "selection": {
            "no_step_within_window": far,
            "frac_no_step": far / max(total_contour, 1),
            "frac_marks_a_step": 1 - far / max(total_contour, 1),
        },
        "localisation_given_a_step": stats(d_near),
        "all_contour_to_step": {"median_incl_far": float(np.median(
            np.where(np.isfinite(d_all), d_all, 1e6))), **stats(d_all)},
        "step_to_contour": stats(d_rev),
    }
    (out_dir / f"summary_{args.dataset}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    s = summary["selection"]
    L = summary["localisation_given_a_step"]
    print(f"\nContour error decomposition  {args.dataset}  n={n} images  "
          f"t={args.t}%  window={args.window}px")
    print(f"\ncontour pixels examined: {total_contour}")
    print(f"  marking no depth step within {args.window:.0f}px : "
          f"{s['no_step_within_window']} ({s['frac_no_step']*100:.1f}%)   <- SELECTION")
    print(f"  marking a real step                  : "
          f"{total_contour - s['no_step_within_window']} ({s['frac_marks_a_step']*100:.1f}%)")
    print(f"\nLOCALISATION, on the contour pixels that do mark a step:")
    print(f"  median {L['median']:.2f}px   p90 {L['p90']:.1f}px   "
          f"<=1px {L['within_1px']*100:.1f}%   <=2px {L['within_2px']*100:.1f}%   "
          f"<=4px {L['within_4px']*100:.1f}%")
    R = summary["step_to_contour"]
    print(f"\nreverse, true step -> nearest contour: median {R['median']:.2f}px   "
          f"<=2px {R['within_2px']*100:.1f}%")
    print(f"\nSaved: {out_dir / f'summary_{args.dataset}.json'}")


if __name__ == "__main__":
    main()
