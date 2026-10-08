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

Two modes, because the object route has a ceiling that segmenting everything does not:

  prompted   YOLO's boxes prompt SAM, one mask per detection. 1.66 masks a frame, covering
             25.2% of the true depth discontinuities - and coverage is what closed every
             object-based direction here, since most of a room's depth edges are walls,
             floors and furniture interiors that COCO's 27 classes never name.
  automatic  SAM's own point grid segments the whole image, ~73 regions a frame, with no
             detector involved. Viable only because sharpening was measured to need no
             precision at all: twice as much spurious contour as real costs 0.8 points of
             314, since refilling a flat region from its own side changes nothing. Over-
             segmentation is therefore free, and coverage is the axis that is not.

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
    p.add_argument(
        "--mode",
        choices=["prompted", "automatic"],
        default="prompted",
        help="prompted: one mask per YOLO box. automatic: segment the whole scene.",
    )
    p.add_argument(
        "--points_per_crop",
        type=int,
        default=32,
        help="Automatic mode's prompt grid. 32 gives ~73 regions at 1.47 s/frame, 16 ~50 at 0.71.",
    )
    p.add_argument("--points_per_batch", type=int, default=64)
    p.add_argument(
        "--pred_iou_thresh",
        type=float,
        default=0.88,
        help="SAM's own quality filter. Lower admits more regions, which costs nothing "
             "on precision but widens the band.",
    )
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
    p.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Enumerate every image under this root instead of using --dataset's naming "
             "convention. For evaluation sets whose RGB is not named rgb_*.png - ScanNet "
             "stores color/*.jpg - staged into an RGB-only tree. Cache keys are relative "
             "to this root, so point the evaluator's --rgb_dir at the same path.",
    )
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


def contour_union(masks, shape, width: int = 1) -> np.ndarray:
    """Union of every region's boundary, not of the regions themselves.

    Automatic mode tiles the image, so adjacent regions share a boundary. Unioning the
    REGIONS would fill the frame and erase every internal edge - the opposite of what the
    sharpening reads. Unioning their CONTOURS keeps each shared edge as a line.
    """
    import cv2

    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    out = np.zeros(shape, bool)
    for m in masks:
        u = np.asarray(m, dtype=np.uint8)
        if u.shape != shape:
            continue
        out |= cv2.dilate(u, k).astype(bool) & ~cv2.erode(u, k).astype(bool)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    from transformers import SamModel, SamProcessor

    rgb_dir = Path(args.rgb_dir)
    out_dir = Path(args.out_dir)
    detail_root = Path(args.detail_artifacts_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half_precision else torch.float32

    logging.info("Loading %s (%s mode)", args.model, args.mode)
    generator = None
    processor = model = None
    if args.mode == "automatic":
        from transformers import pipeline

        # fp32 deliberately: the mask NMS raises "dets should have the same type as
        # scores" under fp16
        generator = pipeline("mask-generation", model=args.model,
                             device=0 if device.type == "cuda" else -1)
    else:
        processor = SamProcessor.from_pretrained(args.model)
        model = SamModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    if args.image_root:
        rgb_dir = Path(args.image_root)
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        paths = sorted(p for p in rgb_dir.rglob("*") if p.suffix.lower() in exts)
        if not paths:
            raise ValueError(f"--image_root {rgb_dir} holds no images")
    elif args.dataset == "hypersim":
        rgb_dir = Path(args.hypersim_root)
        paths = sorted(rgb_dir.rglob("rgb_*.png"))
    else:
        paths = [p for p, _ in list_nyu_pairs(rgb_dir)]
    if args.max_images:
        paths = paths[: args.max_images]

    n_written = n_empty = 0
    n_regions, band_frac = [], []
    for rgb_path in tqdm(paths, desc=f"sam[{args.dataset}]"):
        dst = _cache_path(rgb_path, rgb_dir, out_dir, "_seg.npz")
        if dst.is_file() and not args.overwrite:
            continue
        image = Image.open(rgb_path).convert("RGB")
        w, h = image.size
        dst.parent.mkdir(parents=True, exist_ok=True)

        if args.mode == "automatic":
            out = generator(image, points_per_batch=args.points_per_batch,
                            points_per_crop=args.points_per_crop,
                            pred_iou_thresh=args.pred_iou_thresh)
            regions = out["masks"]
            n_regions.append(len(regions))
            if not regions:
                n_empty += 1
                np.savez_compressed(dst, packed=np.zeros((0, 0), np.uint8), n=0)
                continue
            # stored as a contour plane, so downstream contour_of() on it is a no-op
            # widening rather than a boundary extraction
            c = contour_union(regions, (h, w))
            band_frac.append(float(c.mean()))
            np.savez_compressed(dst, packed=np.packbits(c.reshape(1, -1), axis=-1), n=1)
            n_written += 1
            continue

        dets = [d for d in load_detections(rgb_path, detail_root)
                if d.score >= args.detection_score_thr]
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

    if n_regions:
        logging.info("automatic: %.1f regions/frame, contour covers %.2f%% of pixels",
                     float(np.mean(n_regions)), float(np.mean(band_frac)) * 100)
    logging.info("wrote %d mask files (%d frames had no detections)", n_written, n_empty)
    logging.info("point --mask_cache_dir at %s", out_dir)


if __name__ == "__main__":
    main()
