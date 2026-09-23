#!/usr/bin/env python
"""Is there enough signal in the image to tell a real depth boundary from a texture edge?

docs/marigold_v2_transfer_results.md measured the wall: 79.8% of SAM's contour pixels
sit where no depth step exists, and discarding them with GT lifts the raw BF1 effect from
-1.90% to +43.39% without moving a single contour. So the direction now needs a selector -
but before training one, this asks whether the information it would need is present at all.

Each contour pixel is labelled "within k px of a true depth discontinuity", and each
candidate feature is scored by AUC against that label. No training, no sharpening: a
feature that cannot separate the two classes here will not separate them inside a model.

The features are deliberately cheap and local, computed from what a selector would have at
inference time - the RGB image, the model's own predicted depth, and the contour mask.
Gradients are taken on LOG depth because BF1 tests a depth RATIO, so a step of the same
relative size should score the same near and far.

A combined score is fit on one half of the frames and scored on the other, so the
combination number is not read off the data it was fit to.
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
from eval_object_oracle_ceiling import _cache_path
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from utils.align_space import add_align_space_arg, get_aligner


def parse_args():
    p = argparse.ArgumentParser(description="AUC of candidate contour-selection features.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/"
                "nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/marigold_v2_pred")
    p.add_argument("--processing_res", type=int, default=640)
    p.add_argument("--mask_cache_dir", type=str,
                   default="D:/lotus/data/oracle_cache/sam_auto48")
    p.add_argument("--output_dir", type=str, default="output/eval_contour_feature_auc")
    p.add_argument("--t", type=float, default=10.0, help="Depth-step threshold in percent.")
    p.add_argument("--label_px", type=float, default=1.0,
                   help="A contour pixel is positive within this far of a true step.")
    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--max_px", type=int, default=2_000_000, help="Subsample cap for scoring.")
    p.add_argument("--seed", type=int, default=42)
    add_align_space_arg(p)
    return p.parse_args()


def load_contour(path: Path, h: int, w: int) -> np.ndarray:
    d = np.load(path)
    n = int(d["n"])
    if n == 0:
        return np.zeros((h, w), bool)
    return np.unpackbits(d["packed"], axis=-1)[:, : h * w].reshape(n, h, w).astype(bool).any(0)


def features(depth: np.ndarray, grey: np.ndarray, contour: np.ndarray) -> dict:
    """Local descriptors of "a depth step passes through here", from inference-time data."""
    log_d = np.log(np.clip(depth, 1e-3, None)).astype(np.float32)
    gx = cv2.Sobel(log_d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_d, cv2.CV_32F, 0, 1, ksize=3)
    dgrad = np.hypot(gx, gy)

    g = grey.astype(np.float32)
    rgrad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                     cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

    def rng(a, k):  # local max - min: a step shows up as spread, a smooth ramp does not
        kern = np.ones((k, k), np.uint8)
        return cv2.dilate(a, kern) - cv2.erode(a, kern)

    def box(a, k):
        return cv2.blur(a, (k, k), borderType=cv2.BORDER_REFLECT)

    mean5 = box(log_d, 5)
    return {
        # the model's own gradient: if it already sees the step, however blurred
        "dgrad": dgrad,
        "dgrad_max5": cv2.dilate(dgrad, np.ones((5, 5), np.uint8)),
        # spread of log depth nearby - large only where the surface actually jumps
        "drange5": rng(log_d, 5),
        "drange9": rng(log_d, 9),
        "dstd5": np.sqrt(np.maximum(box(log_d * log_d, 5) - mean5 * mean5, 0.0)),
        # colour edge: correlated with depth edges, but fires on texture too
        "rgrad": rgrad,
        # texture regions are dense with contours; object outlines are not
        "cdens15": box(contour.astype(np.float32), 15),
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC; equals P(score of a positive > score of a negative)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks within ties so constant features score 0.5 rather than something else
    s_sorted = scores[order]
    start = 0
    for i in range(1, s_sorted.size + 1):
        if i == s_sorted.size or s_sorted[i] != s_sorted[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = Path(args.rgb_dir)
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    align = get_aligner(args.align_space)
    rng_ = np.random.default_rng(args.seed)

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    cols, labels, halves = {}, [], []
    for idx, (rgb_path, depth_path) in enumerate(tqdm(pairs, desc="features")):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        if valid.sum() < 100:
            continue
        pp = _cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        mp = _cache_path(rgb_path, rgb_dir, Path(args.mask_cache_dir), "_seg.npz")
        if not (pp.is_file() and mp.is_file()):
            raise FileNotFoundError(f"missing cache for {rgb_path}")
        depth = align(np.load(pp).astype(np.float64), gt, valid)
        if depth is None:
            continue
        contour = load_contour(mp, h, w) & valid
        if not contour.any():
            continue

        grey = np.array(Image.open(rgb_path).convert("L"))
        dist = ndi.distance_transform_edt(~discontinuities(gt, valid, args.t))
        y = (dist <= args.label_px)[contour]

        for k, v in features(depth, grey, contour).items():
            cols.setdefault(k, []).append(v[contour].astype(np.float32))
        labels.append(y)
        halves.append(np.full(y.size, idx % 2, np.int8))  # frame-level split, not pixel-level

    X = {k: np.concatenate(v) for k, v in cols.items()}
    y = np.concatenate(labels)
    half = np.concatenate(halves)
    n = y.size
    if n > args.max_px:
        sel = rng_.choice(n, size=args.max_px, replace=False)
        X = {k: v[sel] for k, v in X.items()}
        y, half = y[sel], half[sel]

    per_feature = {}
    for k, v in X.items():
        a = auc(v.astype(np.float64), y)
        per_feature[k] = max(a, 1 - a)  # a feature that anti-correlates is just as usable
        per_feature[k + "__raw"] = a

    # combination: logistic regression fit on half the FRAMES, scored on the other half
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    names = sorted(X)
    M = np.stack([X[k] for k in names], axis=1).astype(np.float64)
    tr, te = half == 0, half == 1
    sc = StandardScaler().fit(M[tr])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(M[tr]), y[tr])
    combined = auc(clf.decision_function(sc.transform(M[te])), y[te])

    summary = {
        "n_images": len(labels), "n_px_scored": int(y.size),
        "positive_rate": float(y.mean()), "label_px": args.label_px, "t": args.t,
        "per_feature_auc": {k: per_feature[k] for k in names},
        "combined_auc_heldout_frames": combined,
        "coefficients": dict(zip(names, clf.coef_[0].tolist())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n輪郭画素 {y.size:,}  正例率 {y.mean()*100:.1f}%  "
          f"（GT の段差から {args.label_px:.0f}px 以内）  n={len(labels)} 枚\n")
    print(f"{'feature':<12}{'AUC':>8}")
    print("-" * 20)
    for k in sorted(names, key=lambda k: -per_feature[k]):
        print(f"{k:<12}{per_feature[k]:>8.4f}")
    print(f"\n{'組み合わせ（別フレームで評価）':<12} {combined:.4f}")
    g = "通過" if (max(per_feature[k] for k in names) >= 0.65 or combined >= 0.75) else "不通過"
    print(f"ゲート（単独 >=0.65 または組み合わせ >=0.75）: {g}")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
