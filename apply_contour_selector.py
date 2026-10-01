#!/usr/bin/env python
"""Run a trained contour selector over a split and write the contours it keeps.

Output goes in the same packed-bits npz layout the sharpening evaluator already reads, so
`eval_yolo_contour_sharpening.py --mask_cache_dir <out_dir> --masks_are_contours` scores the
selection with no change to the scoring path - the same path that produced every other
number in docs/contour_selector_results.md.

Two details have to match training or the selector sees different inputs than it learned on:
the prediction is standardised per image (Log-stage2's affine-invariant log depth has no
meaningful level or scale), and retention is set by taking the top share of contour pixels
per image rather than by a fixed probability, because a global threshold would drift with
how many contours a scene happens to have.

`--min_component` is off by default: the component-size floor added for the coherence
constraint measured 3.9 points WORSE than leaving the selector's output alone (+8.61%
against +12.49%), since convolution is already smooth enough.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_contour_feature_auc import load_contour
from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path
from utils.eval_frames import add_dataset_args, list_frames, valid_mask
from train_contour_selector import UNet

NYU = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar"


def parse_args():
    p = argparse.ArgumentParser(description="Apply a trained contour selector.")
    p.add_argument("--checkpoint", type=str, default="output/contour_selector/best.pt")
    p.add_argument("--rgb_dir", type=str, default=f"{NYU}/test")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/marigold_v2_pred/res640")
    p.add_argument("--mask_cache_dir", type=str, default="D:/lotus/data/oracle_cache/sam_auto48")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--pred_space",
        choices=["log", "disparity"],
        default="log",
        help="What the cached prediction holds. 'log' is Marigold V2's affine-invariant log "
             "depth, fed as-is. 'disparity' is Lotus's, converted with -log(d) first: "
             "disparity rises as depth falls, so feeding it raw hands the selector an "
             "inverted signal, and -log(d) is log depth up to an affine the per-image "
             "standardisation removes anyway.",
    )
    p.add_argument("--retention", type=float, default=0.06)
    p.add_argument("--min_component", type=int, default=0,
                   help="Drop kept components smaller than this. 0 disables; measured worse.")
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--label_px", type=float, default=1.0)
    p.add_argument("--report_precision", action="store_true",
                   help="Also score precision against GT - diagnostic only, never an input.")
    p.add_argument(
        "--no_sam",
        action="store_true",
        help="Match the training ablation: no contour channel, and candidates are every "
             "valid pixel instead of the SAM contour. --retention must then be given in "
             "the same ABSOLUTE terms (see docs).",
    )
    add_dataset_args(p)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rgb_dir, out_dir = Path(args.rgb_dir), Path(args.out_dir)
    pred_dir, mask_dir = Path(args.pred_cache_dir), Path(args.mask_cache_dir)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 4 if args.no_sam else 5
    model = UNet(in_ch, ck["args"]["base_ch"]).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"checkpoint: {args.checkpoint}  epoch {ck['epoch']}  "
          f"val 適合率 {ck['val_precision']*100:.1f}%")

    pairs = list_frames(args)
    if args.max_images:
        pairs = pairs[: args.max_images]

    keep_frac, tp, sel_n, pos_n, cont_n = [], 0, 0, 0, 0
    with torch.no_grad():
        for rgb_path, depth_path in tqdm(pairs, desc="apply"):
            gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
            h, w = gt.shape
            valid = valid_mask(args.dataset, gt)
            if args.no_sam:
                contour = valid                      # every valid pixel is a candidate
            else:
                contour = load_contour(
                    _cache_path(rgb_path, rgb_dir, mask_dir, "_seg.npz"), h, w) & valid

            out_path = _cache_path(rgb_path, rgb_dir, out_dir, "_seg.npz")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not contour.any():
                np.savez_compressed(out_path, packed=np.packbits(
                    np.zeros((1, h * w), bool), axis=-1), n=1)
                continue

            pred = np.load(_cache_path(rgb_path, rgb_dir, pred_dir, "_pred.npy")).astype(np.float32)
            if args.pred_space == "disparity":
                pred = -np.log(np.clip(pred, 1e-3, None))       # -> log depth up to affine
            pred = (pred - pred.mean()) / (pred.std() + 1e-6)   # as in training
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"), np.float32) / 127.5 - 1.0
            planes = [rgb.transpose(2, 0, 1), pred[None]]
            if not args.no_sam:
                planes.append(contour.astype(np.float32)[None] * 2.0 - 1.0)
            x = np.concatenate(planes, axis=0)[None]
            score = model(torch.from_numpy(x).to(device))[0, 0].float().cpu().numpy()

            k = max(int(round(int(contour.sum()) * args.retention)), 1)
            thr = np.partition(score[contour], -k)[-k]
            keep = contour & (score >= thr)
            if args.min_component > 0:
                n, lab, st, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8), 8)
                ok = np.zeros(n, bool)
                ok[1:] = st[1:, cv2.CC_STAT_AREA] >= args.min_component
                keep = ok[lab]

            keep_frac.append(keep.sum() / contour.sum())
            np.savez_compressed(out_path, packed=np.packbits(keep.reshape(1, -1), axis=-1), n=1)

            if args.report_precision:
                from scipy import ndimage as ndi
                pos = contour & (ndi.distance_transform_edt(
                    ~discontinuities(gt, valid, args.t)) <= args.label_px)
                tp += int((keep & pos).sum()); sel_n += int(keep.sum())
                pos_n += int(pos.sum()); cont_n += int(contour.sum())

    summary = {"checkpoint": args.checkpoint, "epoch": ck["epoch"],
               "pred_space": args.pred_space,
               "val_precision": ck["val_precision"], "n_images": len(keep_frac),
               "retention": float(np.mean(keep_frac)), "min_component": args.min_component}
    if args.report_precision and sel_n:
        prec = tp / sel_n
        summary.update({"precision": prec, "recall": tp / max(pos_n, 1),
                        "base_rate": pos_n / max(cont_n, 1),
                        "predicted_raw_bf1_gain_pct": 0.69 * (prec * 100 - 37.1)})
        print(f"\n適合率 {prec*100:.1f}%  再現率 {tp/max(pos_n,1)*100:.1f}%  "
              f"ベース {pos_n/max(cont_n,1)*100:.2f}%  "
              f"換算 BF1 {0.69*(prec*100-37.1):+.1f}%")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "apply_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"残存率 {np.mean(keep_frac)*100:.2f}%   {len(keep_frac)} 枚 -> {out_dir}")


if __name__ == "__main__":
    main()
