#!/usr/bin/env python
"""Oracle ceiling measurement for object-region conditioning on NYUv2.

Answers "how much abs_rel could ANY object-region method possibly recover?" by
substituting ground-truth depth into object regions of the official Lotus
prediction, post-hoc. No training, no conditioning - it upper-bounds every
object-conditioning approach at once.

Variants measured (see docs/lotus_improvement_plan.md, Phase 0-2):

  baseline    official Lotus, untouched
  A_bbox      per-object depth LEVEL corrected to GT inside bbox rectangles,
              predicted shape preserved  -> ceiling of a *perfect regressor*
  A_mask      same, inside YOLO-seg instance masks
  B_bbox      object bbox rectangles fully replaced by GT
              -> ceiling of *any* object-region method
  B_mask      object seg masks fully replaced by GT
  BG          the complement (everything outside objects) replaced by GT
              -> the share of the error that lives in the background
  *_ctrl      CONTROL: the same masks circularly translated somewhere unrelated.
              Same area and shape, no object correspondence. Added after the Booster
              run showed this control absorbs over half of an apparently strong
              material oracle - where the model is globally off, substituting GT
              anywhere helps regardless of what the mask means. Only the excess over
              this control is attributable to objects.

B and BG partition the image, so their gains show how the total error splits
between object and non-object regions.

Alignment is applied BEFORE substitution: the prediction is scale/shift-fitted
to GT exactly as in eval_regressor_predepth_nyuv2.py, and only then are object
pixels overwritten. Substituting first would let GT leak into the fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from evaluation.util.metric import abs_relative_difference, delta1_acc
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from pipeline import LotusDPipeline
from utils.object_detection_cache import load_detections
from utils.seed_all import seed_all

VARIANTS = ["baseline", "A_bbox", "A_mask", "B_bbox", "B_mask", "BG",
            "A_bbox_ctrl", "A_mask_ctrl", "B_bbox_ctrl", "B_mask_ctrl"]


def parse_args():
    p = argparse.ArgumentParser(description="Object-region oracle ceiling on NYUv2.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument(
        "--detail_artifacts_dir",
        type=str,
        default="D:/lotus/data/nyuv2_detail_artifacts/test",
        help="Root of cached YOLO detection JSONs (bboxes).",
    )
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--yolo_model", type=str, default="yolov8n-seg.pt")
    p.add_argument(
        "--pred_cache_dir",
        type=str,
        default="D:/lotus/data/oracle_cache/lotus_pred",
        help=(
            "Cache dir for official Lotus disparity predictions (.npy, float16). "
            "processing_res is appended, since predictions differ per resolution."
        ),
    )
    p.add_argument(
        "--mask_cache_dir",
        type=str,
        default="D:/lotus/data/oracle_cache/yolo_seg",
        help="Cache dir for per-image YOLO-seg instance masks (.npz, packed bits).",
    )
    p.add_argument("--output_dir", type=str, default="output/eval_object_oracle_ceiling")
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help=(
            "Lotus inference resolution. 768 is the official default "
            "(pipeline.LotusDPipeline.default_processing_resolution). The historical "
            "experiment series ran at 512 by mistake - see docs/phase0_findings.md."
        ),
    )
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_controls", type=int, default=3,
                   help="Random relocations averaged per control mask.")
    p.add_argument(
        "--min_object_pixels",
        type=int,
        default=50,
        help="Skip per-object level correction for objects with fewer valid GT pixels.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# prediction / mask caches
# --------------------------------------------------------------------------- #


def _cache_path(rgb_path: Path, rgb_dir: Path, cache_dir: Path, suffix: str) -> Path:
    rel = rgb_path.relative_to(rgb_dir)
    return cache_dir / rel.parent / f"{rel.stem}{suffix}"


@torch.no_grad()
def predict_disparity(pipe, rgb_np, timestep, processing_res, generator):
    """Official Lotus forward pass -> [H,W] disparity in prediction space."""
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = (image / 127.5 - 1.0).to(device)
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device_type=device.type)
    with ctx:
        out = pipe(
            rgb_in=image,
            prompt="",
            num_inference_steps=1,
            generator=generator,
            output_type="np",
            timesteps=[timestep],
            task_emb=task_emb,
            processing_res=processing_res,
            match_input_res=True,
        ).images[0]
    return (out.mean(axis=-1) if out.ndim == 3 else out).astype(np.float32)


def load_or_build_pred(rgb_path, rgb_dir, cache_dir, pipe, rgb_np, args, generator):
    path = _cache_path(rgb_path, rgb_dir, cache_dir, "_pred.npy")
    if path.is_file():
        return np.load(path).astype(np.float64)
    pred = predict_disparity(pipe, rgb_np, args.timestep, args.processing_res, generator)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pred.astype(np.float16))
    return pred.astype(np.float64)


def load_or_build_masks(rgb_path, rgb_dir, cache_dir, yolo, rgb_np, score_thr):
    """Return [N,H,W] bool instance masks from YOLO-seg, cached as packed bits."""
    h, w = rgb_np.shape[:2]
    path = _cache_path(rgb_path, rgb_dir, cache_dir, "_seg.npz")
    if path.is_file():
        data = np.load(path)
        packed, n = data["packed"], int(data["n"])
        if n == 0:
            return np.zeros((0, h, w), dtype=bool)
        return np.unpackbits(packed, axis=-1, count=h * w).reshape(n, h, w).astype(bool)

    import cv2

    results = yolo.predict(source=rgb_np, imgsz=640, conf=score_thr, verbose=False)
    result = results[0]
    masks = []
    if result.masks is not None and result.boxes is not None:
        for i, box in enumerate(result.boxes):
            if float(box.conf.item()) < score_thr or i >= len(result.masks.data):
                continue
            seg = result.masks.data[i].detach().cpu().numpy()
            if seg.ndim == 3:
                seg = seg[0]
            seg = cv2.resize(seg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append(seg > 0.5)
    arr = np.stack(masks, axis=0) if masks else np.zeros((0, h, w), dtype=bool)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed=np.packbits(arr.reshape(len(arr), -1), axis=-1) if len(arr) else np.zeros((0, 0), np.uint8),
        n=len(arr),
    )
    return arr


# --------------------------------------------------------------------------- #
# oracle construction
# --------------------------------------------------------------------------- #


def align_to_gt(pred_disp, gt, valid):
    """Scale/shift-fit the prediction to GT and return aligned metric depth."""
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


def oracle_level(depth_base, gt, instance_masks, valid, min_px):
    """Oracle A: correct each object's depth LEVEL, keep its predicted shape.

    A perfect regressor knows one number per object, so the best it can do is
    put each object at the right distance - it cannot fix the object's shape.
    Multiplicative because abs_rel is a relative error.
    """
    out = depth_base.copy()
    for mask in instance_masks:
        m = mask & valid
        if m.sum() < min_px:
            continue
        pred_med = float(np.median(depth_base[m]))
        gt_med = float(np.median(gt[m]))
        if pred_med <= 1e-6 or not np.isfinite(gt_med):
            continue
        out[m] = np.clip(depth_base[m] * (gt_med / pred_med), 1e-3, 10.0)
    return out


def oracle_replace(depth_base, gt, region, valid):
    """Oracle B / BG: hand the region its ground-truth depth outright."""
    out = depth_base.copy()
    m = region & valid
    out[m] = gt[m]
    return out


def bbox_instance_masks(dets, h, w):
    masks = []
    for det in dets:
        x1, y1, x2, y2 = det.bbox
        if x2 <= x1 or y2 <= y1:
            continue
        m = np.zeros((h, w), dtype=bool)
        m[y1:y2, x1:x2] = True
        masks.append(m)
    return masks


def score(depth, gt, valid):
    return (
        float(abs_relative_difference(torch.from_numpy(depth), torch.from_numpy(gt), torch.from_numpy(valid))),
        float(delta1_acc(torch.from_numpy(depth), torch.from_numpy(gt), torch.from_numpy(valid))),
    )


# --------------------------------------------------------------------------- #


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    seed_all(args.seed)

    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_artifacts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Predictions depend on processing_res, so they must not share a cache across
    # resolutions. Masks come from YOLO on the source image and do not.
    pred_cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    mask_cache = Path(args.mask_cache_dir)

    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]
    logging.info("Images: %d", len(pairs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half_precision else torch.float32
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Both models are only built if their cache is cold.
    pipe = yolo = None

    def get_pipe():
        nonlocal pipe
        if pipe is None:
            logging.info("Loading Lotus pipeline: %s", args.core_model)
            pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=dtype).to(device)
        return pipe

    def get_yolo():
        nonlocal yolo
        if yolo is None:
            from ultralytics import YOLO

            logging.info("Loading YOLO-seg: %s", args.yolo_model)
            yolo = YOLO(args.yolo_model)
        return yolo

    rng = np.random.default_rng(args.seed)
    acc = {v: {"absrel": [], "d1": []} for v in VARIANTS}
    rows = []
    n_scored = 0
    n_no_object = 0

    for rgb_path, depth_path in tqdm(pairs, desc="oracle_ceiling"):
        rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = rgb_np.shape[:2]

        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        if valid.sum() < 100:
            continue

        pred_path = _cache_path(rgb_path, rgb_dir, pred_cache, "_pred.npy")
        pred = (
            np.load(pred_path).astype(np.float64)
            if pred_path.is_file()
            else load_or_build_pred(rgb_path, rgb_dir, pred_cache, get_pipe(), rgb_np, args, generator)
        )
        if pred.shape != gt.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.float32)).resize((w, h), Image.BILINEAR),
                dtype=np.float64,
            )

        depth_base = align_to_gt(pred, gt, valid)
        if depth_base is None:
            continue

        dets = [d for d in load_detections(rgb_path, detail_root) if d.score >= args.detection_score_thr]
        bbox_masks = bbox_instance_masks(dets, h, w)

        seg_path = _cache_path(rgb_path, rgb_dir, mask_cache, "_seg.npz")
        seg_masks = list(
            load_or_build_masks(
                rgb_path,
                rgb_dir,
                mask_cache,
                None if seg_path.is_file() else get_yolo(),
                rgb_np,
                args.detection_score_thr,
            )
        )

        h, w = valid.shape
        bbox_union = (
            np.any(np.stack(bbox_masks), axis=0) if bbox_masks else np.zeros((h, w), bool)
        )
        seg_union = np.any(np.stack(seg_masks), axis=0) if seg_masks else np.zeros((h, w), bool)
        if not bbox_masks and not seg_masks:
            n_no_object += 1

        variants = {
            "baseline": depth_base,
            "A_bbox": oracle_level(depth_base, gt, bbox_masks, valid, args.min_object_pixels),
            "A_mask": oracle_level(depth_base, gt, seg_masks, valid, args.min_object_pixels),
            "B_bbox": oracle_replace(depth_base, gt, bbox_union, valid),
            "B_mask": oracle_replace(depth_base, gt, seg_union, valid),
            "BG": oracle_replace(depth_base, gt, ~bbox_union, valid),
        }

        # Controls: identical masks, relocated. Averaged over several offsets so the
        # control is not itself a lucky draw.
        for tag, masks, union in (("bbox", bbox_masks, bbox_union),
                                  ("mask", seg_masks, seg_union)):
            accA = np.zeros_like(depth_base)
            accB = np.zeros_like(depth_base)
            for _ in range(args.n_controls):
                dy = int(rng.integers(h // 5, 4 * h // 5))
                dx = int(rng.integers(w // 5, 4 * w // 5))
                shifted = [np.roll(m, (dy, dx), axis=(0, 1)) for m in masks]
                accA += oracle_level(depth_base, gt, shifted, valid, args.min_object_pixels)
                accB += oracle_replace(depth_base, gt, np.roll(union, (dy, dx), axis=(0, 1)), valid)
            variants[f"A_{tag}_ctrl"] = accA / args.n_controls
            variants[f"B_{tag}_ctrl"] = accB / args.n_controls

        row = {
            "filename": str(rgb_path.relative_to(rgb_dir)).replace("\\", "/"),
            "n_detections": len(dets),
            "n_seg_masks": len(seg_masks),
            "bbox_coverage": float((bbox_union & valid).sum() / max(valid.sum(), 1)),
            "seg_coverage": float((seg_union & valid).sum() / max(valid.sum(), 1)),
        }
        for name, depth in variants.items():
            a, d = score(depth, gt, valid)
            acc[name]["absrel"].append(a)
            acc[name]["d1"].append(d)
            row[f"absrel_{name}"] = a
        rows.append(row)
        n_scored += 1

    if not rows:
        raise RuntimeError("No images scored.")

    base = float(np.mean(acc["baseline"]["absrel"]))
    summary = {
        "num_images": n_scored,
        "num_images_without_objects": n_no_object,
        "detection_score_thr": args.detection_score_thr,
        "processing_res": args.processing_res,
        "core_model": args.core_model,
        "mean_bbox_coverage": float(np.mean([r["bbox_coverage"] for r in rows])),
        "mean_seg_coverage": float(np.mean([r["seg_coverage"] for r in rows])),
        "variants": {},
    }
    for name in VARIANTS:
        a = float(np.mean(acc[name]["absrel"]))
        summary["variants"][name] = {
            "abs_rel": a,
            "delta1": float(np.mean(acc[name]["d1"])),
            "gain_vs_baseline": base - a,
            "gain_pct": 100.0 * (base - a) / base if base > 0 else 0.0,
        }

    summary["net_of_control"] = {}
    for k in ("A_bbox", "A_mask", "B_bbox", "B_mask"):
        summary["net_of_control"][k] = {
            "real": summary["variants"][k]["gain_pct"],
            "control": summary["variants"][k + "_ctrl"]["gain_pct"],
            "net": summary["variants"][k]["gain_pct"] - summary["variants"][k + "_ctrl"]["gain_pct"],
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    label = {
        "baseline": "baseline (official Lotus)",
        "A_bbox": "Oracle A  bbox  (level only)",
        "A_mask": "Oracle A  mask  (level only)",
        "B_bbox": "Oracle B  bbox  (full GT)",
        "B_mask": "Oracle B  mask  (full GT)",
        "BG": "Oracle BG (background full GT)",
        "A_bbox_ctrl": "  control: bbox relocated",
        "A_mask_ctrl": "  control: mask relocated",
        "B_bbox_ctrl": "  control: bbox relocated",
        "B_mask_ctrl": "  control: mask relocated",
    }
    print(f"\n{'variant':<32} {'abs_rel':>9} {'delta1':>8} {'gain':>9} {'gain%':>8}")
    print("-" * 70)
    for name in VARIANTS:
        v = summary["variants"][name]
        print(
            f"{label[name]:<32} {v['abs_rel']:>9.5f} {v['delta1']:>8.4f} "
            f"{v['gain_vs_baseline']:>9.5f} {v['gain_pct']:>7.1f}%"
        )
    print(
        f"\nobject coverage: bbox {summary['mean_bbox_coverage']*100:.1f}% of valid px, "
        f"seg {summary['mean_seg_coverage']*100:.1f}%"
    )
    print(f"images: {n_scored} ({n_no_object} with no detected object)")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
