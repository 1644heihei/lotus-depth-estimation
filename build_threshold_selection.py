#!/usr/bin/env python
"""The zero-training control: keep the contour pixels where a hand-crafted feature is largest.

This is the control the learned selector has to beat. It answers "how much of the gain is
available without training at all?", and §3 of docs/contour_selector_results.md reads the
difference between the two as the learning's contribution.

The feature is `dgrad_max5` - the local maximum of the log-depth gradient magnitude - which
the AUC sweep found to be the best single one (0.876 on NYUv2, output/eval_contour_feature_auc).
Nothing is fitted: the feature is fixed, and the selection is the top share of it per image.

The selection is invariant to the affine the prediction is defined up to. Alignment in log
space is `a * log d + b`; a shift drops out of a gradient, a positive scale multiplies every
pixel's feature by the same factor, and a per-image top-share ranking is unchanged by either.
So this control needs no ground truth, and it gives the same answer whether it is built from
the raw prediction or from one aligned to GT.

Written because the original control was built by an inline script that was not kept, which
made it impossible to run the same control against a second depth model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contour_selector import (PRED_SPACES, cache_path, contour_labels,
                              load_candidates, to_log_depth, top_share_mask)
from utils.align_space import add_align_space_arg, get_aligner
from utils.eval_frames import (add_dataset_args, depth_scale, list_frames,
                               valid_mask)

NYU = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar"


def parse_args():
    p = argparse.ArgumentParser(description="Hand-crafted threshold selection (no training).")
    p.add_argument("--rgb_dir", type=str, default=f"{NYU}/test")
    p.add_argument("--pred_cache_dir", type=str,
                   default="D:/lotus/data/marigold_v2_pred/res640")
    p.add_argument("--mask_cache_dir", type=str,
                   default="D:/lotus/data/oracle_cache/sam_auto48")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pred_space", choices=PRED_SPACES, default="log")
    p.add_argument("--retention", type=float, default=0.06)
    p.add_argument("--feature", choices=["dgrad", "dgrad_max5", "drange5", "drange9"],
                   default="dgrad_max5",
                   help="dgrad_max5 is the best single feature by AUC (0.876 on NYUv2).")
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--label_px", type=float, default=1.0)
    p.add_argument("--report_precision", action="store_true")
    p.add_argument(
        "--align_to_gt",
        action="store_true",
        help="Build the feature from the GT-ALIGNED prediction, as "
             "eval_contour_feature_auc.py does. Kept only to reproduce the original "
             "control, which was built this way. It is not affine-invariant in practice: "
             "the aligner clips to [DEPTH_MIN, DEPTH_MAX], and saturating the far field "
             "flattens the gradient there, which changes the ranking. A selector that "
             "needs GT to decide where to sharpen is not one you could deploy, so the "
             "default is off.",
    )
    add_align_space_arg(p)
    add_dataset_args(p)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def feature_map(log_depth: np.ndarray, name: str) -> np.ndarray:
    """One of the hand-crafted descriptors of "a depth step passes through here".

    Taken on LOG depth because BF1 tests a depth RATIO: a step of the same relative size
    should score the same near and far.
    """
    d = log_depth.astype(np.float32)
    if name in ("dgrad", "dgrad_max5"):
        g = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3),
                     cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))
        return g if name == "dgrad" else cv2.dilate(g, np.ones((5, 5), np.uint8))
    k = np.ones((5, 5) if name == "drange5" else (9, 9), np.uint8)
    return cv2.dilate(d, k) - cv2.erode(d, k)


def main():
    args = parse_args()
    rgb_dir, out_dir = Path(args.rgb_dir), Path(args.out_dir)
    pred_dir, mask_dir = Path(args.pred_cache_dir), Path(args.mask_cache_dir)
    align = get_aligner(args.align_space)
    pairs = list_frames(args)
    if args.max_images:
        pairs = pairs[: args.max_images]

    keep_frac, tp, sel_n, pos_n, cont_n = [], 0, 0, 0, 0
    for rgb_path, depth_path in tqdm(pairs, desc=args.feature):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(args.dataset)
        h, w = gt.shape
        valid = valid_mask(args.dataset, gt)
        candidates = load_candidates(rgb_path, rgb_dir, mask_dir, valid)

        out_path = cache_path(rgb_path, rgb_dir, out_dir, "_seg.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not candidates.any():
            np.savez_compressed(out_path, packed=np.packbits(
                np.zeros((1, h * w), bool), axis=-1), n=1)
            continue

        raw = np.load(cache_path(rgb_path, rgb_dir, pred_dir, "_pred.npy")).astype(np.float32)
        if args.align_to_gt:
            metric = align(raw.astype(np.float64), gt, valid)
            if metric is None:
                continue
            log_d = np.log(np.clip(metric, 1e-3, None))
        else:
            log_d = to_log_depth(raw, args.pred_space)
        score = feature_map(log_d, args.feature)
        keep = top_share_mask(score, candidates, args.retention)

        keep_frac.append(keep.sum() / candidates.sum())
        np.savez_compressed(out_path, packed=np.packbits(keep.reshape(1, -1), axis=-1), n=1)

        if args.report_precision:
            pos = candidates & contour_labels(gt, valid, args.t, args.label_px)
            tp += int((keep & pos).sum()); sel_n += int(keep.sum())
            pos_n += int(pos.sum()); cont_n += int(candidates.sum())

    summary = {"feature": args.feature, "pred_space": args.pred_space,
               "align_to_gt": bool(args.align_to_gt),
               "n_images": len(keep_frac), "retention": float(np.mean(keep_frac))}
    if args.report_precision and sel_n:
        prec = tp / sel_n
        summary.update({"precision": prec, "recall": tp / max(pos_n, 1),
                        "base_rate": pos_n / max(cont_n, 1)})
        print(f"\n適合率 {prec*100:.1f}%  再現率 {tp/max(pos_n,1)*100:.1f}%  "
              f"ベース {pos_n/max(cont_n,1)*100:.2f}%")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "apply_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"残存率 {np.mean(keep_frac)*100:.2f}%   {len(keep_frac)} 枚 -> {out_dir}")


if __name__ == "__main__":
    main()
