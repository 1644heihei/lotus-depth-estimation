#!/usr/bin/env python
"""Run a trained contour selector over a split and write the contours it keeps.

Output goes in the same packed-bits npz layout the sharpening evaluator already reads, so
`eval_yolo_contour_sharpening.py --mask_cache_dir <out_dir> --masks_are_contours` scores the
selection with no change to the scoring path - the same path that produced every other
number in docs/contour_selector_results.md.

The input pipeline is `contour_selector.inputs`, the same module training reads through, so
the two cannot drift: a mismatch in the per-image standardisation or the prediction space
would leave the selector running on magnitudes it never saw, and nothing would fail except
the BF1 at the end.

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

from contour_selector import (PRED_SPACES, UNet, build_input, cache_path,
                              contour_labels, in_channels, load_candidates,
                              load_prediction, top_share_mask)
from utils.eval_frames import (add_dataset_args, depth_scale, list_frames,
                               valid_mask)

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
        choices=PRED_SPACES,
        default="log",
        help="What the cached prediction holds. 'log' is Marigold V2's affine-invariant "
             "log depth, fed as-is. 'disparity' is Lotus's, converted with -log(d) first "
             "(see contour_selector.inputs.to_log_depth).",
    )
    p.add_argument("--retention", type=float, default=0.06)
    p.add_argument("--min_component", type=int, default=0,
                   help="Drop kept components smaller than this. 0 disables; measured worse.")
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--label_px", type=float, default=1.0)
    p.add_argument("--report_precision", action="store_true",
                   help="Also score precision against GT - diagnostic only, never an input.")
    p.add_argument(
        "--no_rgb",
        action="store_true",
        help="Match the training ablation: no RGB channels.",
    )
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


def write_mask(path: Path, keep: np.ndarray) -> None:
    """The packed-bits layout the sharpening evaluator reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=np.packbits(keep.reshape(1, -1), axis=-1), n=1)


def drop_small_components(keep: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8), 8)
    ok = np.zeros(n, bool)
    ok[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return ok[lab]


def main():
    args = parse_args()
    rgb_dir, out_dir = Path(args.rgb_dir), Path(args.out_dir)
    pred_dir, mask_dir = Path(args.pred_cache_dir), Path(args.mask_cache_dir)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels(no_rgb=args.no_rgb, no_sam=args.no_sam),
                 ck["args"]["base_ch"]).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"checkpoint: {args.checkpoint}  epoch {ck['epoch']}  "
          f"val 適合率 {ck['val_precision']*100:.1f}%")

    pairs = list_frames(args)
    if args.max_images:
        pairs = pairs[: args.max_images]

    keep_frac, tp, sel_n, pos_n, cont_n = [], 0, 0, 0, 0
    with torch.no_grad():
        for rgb_path, depth_path in tqdm(pairs, desc="apply"):
            gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(args.dataset)
            h, w = gt.shape
            valid = valid_mask(args.dataset, gt)
            candidates = load_candidates(rgb_path, rgb_dir, mask_dir, valid,
                                         no_sam=args.no_sam)

            out_path = cache_path(rgb_path, rgb_dir, out_dir, "_seg.npz")
            if not candidates.any():
                write_mask(out_path, np.zeros((h, w), bool))
                continue

            pred = load_prediction(rgb_path, rgb_dir, pred_dir, args.pred_space)
            x = build_input(rgb_path, pred, candidates,
                            no_rgb=args.no_rgb, no_sam=args.no_sam)[None]
            score = model(torch.from_numpy(x).to(device))[0, 0].float().cpu().numpy()

            keep = top_share_mask(score, candidates, args.retention)
            if args.min_component > 0:
                keep = drop_small_components(keep, args.min_component)

            keep_frac.append(keep.sum() / candidates.sum())
            write_mask(out_path, keep)

            if args.report_precision:
                pos = candidates & contour_labels(gt, valid, args.t, args.label_px)
                tp += int((keep & pos).sum()); sel_n += int(keep.sum())
                pos_n += int(pos.sum()); cont_n += int(candidates.sum())

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
