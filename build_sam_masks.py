#!/usr/bin/env python
"""Regenerate object masks with HQ-SAM, prompted by the YOLO boxes already cached.

docs/contour_sharpening_findings.md leaves one number between this investigation and a
working method. Treating a contour as a barrier and refilling each side from its own side
is worth +279.9% BF1 net of control, with no training and abs_rel improving - but the gain
ends at 1.71px of contour error, and YOLO-seg sits at 2.00px where a depth step exists.
0.3px.

YOLO's masks come from prototypes at stride 4 of a 640px input, upsampled to full size, so
their coarseness is structural rather than a tuning failure. HQ-SAM adds a high-quality
output token to SAM's decoder specifically to sharpen boundary predictions, and takes boxes
as prompts - so the detections stay exactly as they are and only the mask changes.

Whether that transfers is genuinely open: SAM improves SEMANTIC boundary quality, and what
is needed here is the location of a DEPTH discontinuity. The two coincide at occluding
contours and part ways elsewhere, so it is measured rather than assumed.

HQ-SAM was the first choice and does not work. All three syscv-community checkpoints in
transformers 5.12.1 return masks that ignore the box prompt - 16-26% of the mask lands
inside the box it was prompted with, against 100% for plain SAM on the same image, and
passing intermediate_embeddings explicitly changes nothing. Plain SAM is what HQ-SAM builds
on, so if its contours miss 1.71px, HQ-SAM's increment would not have closed the gap
either.

Writes the same packed-bits .npz layout eval_object_oracle_ceiling.load_or_build_masks
reads, so every downstream script works unchanged by pointing --mask_cache_dir here.
"""

from __future__ import annotations

import argparse
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

from eval_object_oracle_ceiling import _cache_path
from eval_regressor_predepth_nyuv2 import list_nyu_pairs
from utils.object_detection_cache import load_detections


def parse_args():
    p = argparse.ArgumentParser(description="HQ-SAM masks from cached YOLO boxes.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--detail_artifacts_dir", type=str, default="D:/lotus/data/nyuv2_detail_artifacts/test")
    p.add_argument("--dataset", choices=["nyuv2", "hypersim"], default="nyuv2")
    p.add_argument("--hypersim_root", type=str, default="D:/lotus/data/hypersim_processed/train")
    p.add_argument("--out_dir", type=str, default="D:/lotus/data/oracle_cache/sam_seg")
    p.add_argument(
        "--union_only",
        action="store_true",
        help=(
            "Store the union of the instance masks as one plane instead of one per object. "
            "Contour sharpening only ever uses the union, and Hypersim is 59k frames."
        ),
    )
    p.add_argument("--model", type=str, default="facebook/sam-vit-huge")
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@torch.no_grad()
def masks_for(model, processor, image, boxes, device, dtype):
    """One mask per box, at the image's own resolution."""
    inputs = processor(image, input_boxes=[[list(map(float, b)) for b in boxes]],
                       return_tensors="pt").to(device)
    if dtype == torch.float16:
        inputs["pixel_values"] = inputs["pixel_values"].half()
    out = model(**inputs, multimask_output=False)
    m = processor.post_process_masks(
        out.pred_masks.float().cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    return m.squeeze(1).numpy().astype(bool)  # [K, H, W]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    from transformers import SamModel, SamProcessor

    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.out_dir)
    detail_root = Path(args.detail_artifacts_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half_precision else torch.float32

    logging.info("Loading %s", args.model)
    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    if args.dataset == "hypersim":
        rgb_dir = Path(args.hypersim_root)
        paths = sorted(rgb_dir.rglob("rgb_*.png"))
    else:
        paths = [p for p, _ in list_nyu_pairs(rgb_dir)]
    if args.max_images:
        paths = paths[: args.max_images]

    n_written = n_empty = 0
    for rgb_path in tqdm(paths, desc=f"sam[{args.dataset}]"):
        dst = _cache_path(rgb_path, rgb_dir, out_dir, "_seg.npz")
        if dst.is_file() and not args.overwrite:
            continue
        image = Image.open(rgb_path).convert("RGB")
        w, h = image.size
        dets = [d for d in load_detections(rgb_path, detail_root)
                if d.score >= args.detection_score_thr]
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not dets:
            n_empty += 1
            # same layout the loader expects for "no objects here"
            np.savez_compressed(dst, packed=np.zeros((0, 0), np.uint8), n=0)
            continue

        m = masks_for(model, processor, image, [d.bbox for d in dets], device, dtype)
        if m.ndim == 2:
            m = m[None]
        if args.union_only:
            m = np.any(m, axis=0)[None]
        packed = np.packbits(m.reshape(len(m), -1), axis=-1)
        np.savez_compressed(dst, packed=packed, n=len(m))
        n_written += 1

    logging.info("wrote %d mask files (%d frames had no detections)", n_written, n_empty)
    logging.info("point --mask_cache_dir at %s", out_dir)


if __name__ == "__main__":
    main()
