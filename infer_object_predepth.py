"""Object pre-depth inference: YOLO bbox detect -> crop -> depth -> fuse.

Usage:
  python infer_object_predepth.py \\
      --pretrained_model_name_or_path jingheya/lotus-depth-d-v2-0-disparity \\
      --input_dir path/to/images \\
      --output_dir output/object_predepth
"""

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers.utils import check_min_version
from tqdm.auto import tqdm

from pipeline import LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.object_detection import YoloBboxDetector, bboxes_to_roi_mask, parse_label_filter
from utils.roi_fusion import fuse_roi_depth
from utils.seed_all import seed_all
from utils.semantic_mask_utils import expand_box, load_mask_for_image

check_min_version("0.28.0.dev0")


def parse_args():
    p = argparse.ArgumentParser(description="BBox-detect, crop, and fuse object depths.")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--detector", type=str, default="yolo", choices=["yolo", "mask"])
    p.add_argument("--roi_mask_dir", type=str, default=None, help="Only for --detector mask.")
    p.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    p.add_argument("--yolo_score_thr", type=float, default=0.25)
    p.add_argument("--yolo_labels", type=str, default="")
    p.add_argument("--yolo_device", type=str, default="0")
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--half_precision", action="store_true", default=True)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--disparity", action="store_true", default=True)
    p.add_argument("--roi_min_area", type=int, default=500)
    p.add_argument("--roi_expand_ratio", type=float, default=0.25)
    p.add_argument("--align_mode", type=str, default="lstsq", choices=["lstsq", "none"])
    p.add_argument("--blend_blur_ksize", type=int, default=31)
    p.add_argument("--fusion_weight", type=float, default=1.0)
    p.add_argument("--save_object_crops", action="store_true")
    return p.parse_args()


def run_depth(pipe, rgb_np: np.ndarray, args, generator):
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = (image / 127.5 - 1.0).to(device)
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    autocast_ctx = nullcontext() if device.type == "mps" else torch.autocast(device_type=device.type)
    with autocast_ctx:
        pred = pipe(
            rgb_in=image,
            prompt="",
            num_inference_steps=1,
            generator=generator,
            output_type="np",
            timesteps=[args.timestep],
            task_emb=task_emb,
            processing_res=args.processing_res,
            match_input_res=True,
        ).images[0]
    return pred.mean(axis=-1)


def detect_boxes(rgb_np, args, bbox_detector, image_path):
    h, w = rgb_np.shape[:2]
    if args.detector == "yolo":
        return bbox_detector.detect(rgb_np, min_area=args.roi_min_area), None

    roi_mask = load_mask_for_image(str(image_path), args.roi_mask_dir)
    if roi_mask is None:
        return [], np.zeros((h, w), dtype=np.float32)
    if roi_mask.shape != (h, w):
        roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    from utils.semantic_mask_utils import connected_component_boxes

    return connected_component_boxes(roi_mask, threshold=0.5, min_area=args.roi_min_area), roi_mask


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.seed is not None:
        seed_all(args.seed)

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = LotusDPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    bbox_detector = None
    if args.detector == "yolo":
        bbox_detector = YoloBboxDetector(
            model_path=args.yolo_model,
            score_thr=args.yolo_score_thr,
            labels_keep=parse_label_filter(args.yolo_labels),
            device=args.yolo_device,
        )

    images = sorted(list(Path(args.input_dir).rglob("*.png")) + list(Path(args.input_dir).rglob("*.jpg")))
    out_predepth = os.path.join(args.output_dir, "pre_depth")
    out_global = os.path.join(args.output_dir, "depth_global")
    out_vis = os.path.join(args.output_dir, "pre_depth_vis")
    out_crops = os.path.join(args.output_dir, "object_crops")
    for d in (out_predepth, out_global, out_vis):
        os.makedirs(d, exist_ok=True)
    if args.save_object_crops:
        os.makedirs(out_crops, exist_ok=True)

    for image_path in tqdm(images, desc="BBox crop pre-depth"):
        rgb_np = np.array(Image.open(image_path).convert("RGB"))
        h, w = rgb_np.shape[:2]
        stem = image_path.stem

        global_depth = run_depth(pipe, rgb_np, args, generator)
        raw_boxes, seg_mask = detect_boxes(rgb_np, args, bbox_detector, image_path)

        crop_preds = []
        for i, box in enumerate(raw_boxes):
            x1, y1, x2, y2 = expand_box(box, (h, w), expand_ratio=args.roi_expand_ratio)
            crop = rgb_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if args.save_object_crops:
                Image.fromarray(crop).save(os.path.join(out_crops, f"{stem}_bbox{i:02d}.jpg"))
            obj_depth = run_depth(pipe, crop, args, generator)
            crop_preds.append(((x1, y1, x2, y2), obj_depth))

        fusion_mask = seg_mask if args.detector == "mask" else None
        pre_depth, _ = fuse_roi_depth(
            global_depth=global_depth,
            crop_preds=crop_preds,
            roi_mask=fusion_mask,
            align_mode=args.align_mode,
            blend_blur_ksize=args.blend_blur_ksize,
            fusion_weight=args.fusion_weight,
        )

        np.save(os.path.join(out_predepth, f"{stem}.npy"), pre_depth)
        np.save(os.path.join(out_global, f"{stem}.npy"), global_depth)
        colorize_depth_map(pre_depth, reverse_color=args.disparity).save(os.path.join(out_vis, f"{stem}.png"))

    logging.info("Done. %d images -> %s", len(images), args.output_dir)


if __name__ == "__main__":
    main()
