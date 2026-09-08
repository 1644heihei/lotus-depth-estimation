#!/usr/bin/env python
"""Draw the contours the sharpening measurements were arguing about.

The numbers say YOLO-seg sits 2.24px from a true depth discontinuity, SAM 2.00px, and the
gain ends at 1.71px - differences small enough that it is worth seeing what they look like.
Each row overlays, on the same frame: the discontinuities ground truth actually contains,
what YOLO-seg traces, what SAM traces given the same boxes, and where Lotus itself already
puts a step. The last panel puts them together on a crop, which is where the sub-pixel
argument becomes visible.

Frames are chosen for the number of detections rather than at random, so the panels have
something in them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path, align_to_gt, load_or_build_masks
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from utils.object_detection_cache import load_detections

GT_C = (0.95, 0.15, 0.15)
YOLO_C = (0.20, 0.55, 1.00)
SAM_C = (0.15, 0.85, 0.35)
LOTUS_C = (1.00, 0.75, 0.10)


def parse_args():
    p = argparse.ArgumentParser(description="Visual comparison of contour sources.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--detail_artifacts_dir", type=str, default="D:/lotus/data/nyuv2_detail_artifacts/test")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/oracle_cache/lotus_pred")
    p.add_argument("--yolo_masks", type=str, default="D:/lotus/data/oracle_cache/yolo_seg")
    p.add_argument("--sam_masks", type=str, default="D:/lotus/data/oracle_cache/sam_seg")
    p.add_argument("--out", type=str, default="output/viz/contour_comparison.png")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--n_rows", type=int, default=4)
    p.add_argument("--zoom", type=int, default=140, help="Side of the crop, in pixels.")
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    return p.parse_args()


def contour_of(mask):
    k = np.ones((3, 3), np.uint8)
    u = mask.astype(np.uint8)
    return cv2.dilate(u, k).astype(bool) & ~cv2.erode(u, k).astype(bool)


def overlay(rgb, layers, alpha=1.0):
    """Paint each (mask, colour) onto a dimmed copy of the frame."""
    out = rgb.astype(np.float32) / 255.0 * 0.65 + 0.35 * 0.15
    for mask, colour in layers:
        m = cv2.dilate(mask.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        out[m] = (1 - alpha) * out[m] + alpha * np.array(colour, np.float32)
    return np.clip(out, 0, 1)


def pick_crop(masks, shape, side):
    """A window centred on the busiest part of the contours, clipped to the frame."""
    h, w = shape
    acc = np.zeros((h, w), np.float32)
    for m in masks:
        acc += m.astype(np.float32)
    acc = cv2.blur(acc, (side // 2 | 1, side // 2 | 1))
    cy, cx = np.unravel_index(int(acc.argmax()), acc.shape)
    y0 = int(np.clip(cy - side // 2, 0, max(h - side, 0)))
    x0 = int(np.clip(cx - side // 2, 0, max(w - side, 0)))
    return y0, x0


def main():
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_artifacts_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"

    # busiest frames first: an empty panel shows nothing about contour quality
    cand = []
    for rgb_path, depth_path in list_nyu_pairs(rgb_dir):
        n = len([d for d in load_detections(rgb_path, detail_root)
                 if d.score >= args.detection_score_thr])
        if n:
            cand.append((n, rgb_path, depth_path))
    cand.sort(key=lambda x: -x[0])
    rows = cand[: args.n_rows]

    fig, axes = plt.subplots(len(rows), 5, figsize=(21, 4.0 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]

    for r, (n_det, rgb_path, depth_path) in enumerate(rows):
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)

        disc = discontinuities(gt, valid, args.t)
        dummy = np.empty((h, w, 3), np.uint8)
        yolo = list(load_or_build_masks(rgb_path, rgb_dir, Path(args.yolo_masks), None,
                                        dummy, args.detection_score_thr))
        sam = list(load_or_build_masks(rgb_path, rgb_dir, Path(args.sam_masks), None,
                                       dummy, args.detection_score_thr))
        yc = contour_of(np.any(np.stack(yolo), axis=0)) & valid if yolo else np.zeros_like(disc)
        sc = contour_of(np.any(np.stack(sam), axis=0)) & valid if sam else np.zeros_like(disc)

        pp = _cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        lot = np.zeros_like(disc)
        if pp.is_file():
            base = align_to_gt(np.load(pp).astype(np.float64), gt, valid)
            if base is not None:
                lot = discontinuities(base, valid, args.t)

        panels = [
            (rgb.astype(np.float32) / 255.0, f"{rgb_path.name}   {n_det} detections"),
            (overlay(rgb, [(disc, GT_C)]), "GT depth discontinuities"),
            (overlay(rgb, [(yc, YOLO_C)]), "YOLO-seg contour  (2.24px)"),
            (overlay(rgb, [(sc, SAM_C)]), "SAM contour  (2.00px)"),
        ]
        for c, (img, title) in enumerate(panels):
            axes[r, c].imshow(img)
            axes[r, c].set_title(title, fontsize=10)
            axes[r, c].axis("off")

        y0, x0 = pick_crop([disc, yc, sc], (h, w), args.zoom)
        s = slice(y0, y0 + args.zoom), slice(x0, x0 + args.zoom)
        crop = overlay(rgb[s], [(lot[s], LOTUS_C), (yc[s], YOLO_C),
                                (sc[s], SAM_C), (disc[s], GT_C)], alpha=0.9)
        axes[r, 4].imshow(cv2.resize(crop, (args.zoom * 3, args.zoom * 3),
                                     interpolation=cv2.INTER_NEAREST))
        axes[r, 4].set_title("all four, 3x crop", fontsize=10)
        axes[r, 4].axis("off")

    fig.legend(
        handles=[mpatches.Patch(color=GT_C, label="GT depth discontinuity (the target)"),
                 mpatches.Patch(color=YOLO_C, label="YOLO-seg contour"),
                 mpatches.Patch(color=SAM_C, label="SAM contour"),
                 mpatches.Patch(color=LOTUS_C, label="Lotus's own discontinuities")],
        loc="lower center", ncol=4, fontsize=11, frameon=False,
    )
    fig.suptitle(
        "Where each source puts a boundary. Sharpening needs the contour within 1.71px of "
        "the red line;\nYOLO manages 2.24px and SAM 2.00px where a depth step exists at all.",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
