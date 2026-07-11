import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from diffusers.utils import check_min_version
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from tqdm.auto import tqdm

from pipeline import LotusDPipeline, LotusGPipeline
from utils.image_utils import colorize_depth_map
from utils.roi_fusion import fuse_roi_depth
from utils.seed_all import seed_all
from utils.semantic_mask_utils import (
    connected_component_boxes,
    expand_box,
    load_mask_for_image,
)

check_min_version("0.28.0.dev0")

COCO_NAMES: Dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck",
    9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def parse_args():
    parser = argparse.ArgumentParser(description="ROI fusion depth inference with Lotus.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="regression", choices=["regression", "generation"])
    parser.add_argument("--task_name", type=str, default="depth")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--processing_res", type=int, default=None)
    parser.add_argument("--output_processing_res", action="store_true")
    parser.add_argument("--resample_method", choices=["bilinear", "bicubic", "nearest"], default="bilinear")
    parser.add_argument("--disparity", action="store_true")

    parser.add_argument("--roi_mask_dir", type=str, default=None)
    parser.add_argument("--detector_backend", type=str, default="none", choices=["none", "coco"])
    parser.add_argument("--detector_score_thr", type=float, default=0.5)
    parser.add_argument("--detector_labels", type=str, default="tv,laptop,book,window,mirror")

    parser.add_argument("--roi_min_area", type=int, default=500)
    parser.add_argument("--roi_expand_ratio", type=float, default=0.25)
    parser.add_argument("--fusion_weight", type=float, default=1.0)
    parser.add_argument("--blend_blur_ksize", type=int, default=31)
    parser.add_argument(
        "--align_mode",
        type=str,
        default="lstsq",
        choices=["lstsq", "none"],
        help="lstsq: scale-shift fit each crop to the global prediction before pasting; none: paste raw crops.",
    )
    return parser.parse_args()


def init_pipeline(args, dtype):
    if args.mode == "generation":
        pipe = LotusGPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype)
    else:
        pipe = LotusDPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype)
    return pipe


def init_coco_detector(device: torch.device):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()
    return model, weights.transforms()


def make_roi_mask_by_detector(
    rgb_np: np.ndarray,
    detector,
    preprocess,
    labels_keep: List[str],
    score_thr: float,
) -> np.ndarray:
    image_tensor = preprocess(Image.fromarray(rgb_np)).to(next(detector.parameters()).device)
    with torch.no_grad():
        outputs = detector([image_tensor])[0]
    mask = np.zeros(rgb_np.shape[:2], dtype=np.float32)
    for box, score, label_idx in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if score.item() < score_thr:
            continue
        label_name = COCO_NAMES.get(int(label_idx.item()), "")
        if labels_keep and label_name not in labels_keep:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)
        mask[y1:y2, x1:x2] = 1.0
    return mask


def run_depth(pipe, rgb_np: np.ndarray, args, generator):
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = image / 127.5 - 1.0
    image = image.to(device)
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type=pipe.device.type)
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
            match_input_res=not args.output_processing_res,
            resample_method=args.resample_method,
        ).images[0]
    return pred.mean(axis=-1)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.seed is not None:
        seed_all(args.seed)

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = init_pipeline(args, dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    detector = None
    preprocess = None
    labels_keep = [x.strip() for x in args.detector_labels.split(",") if x.strip()]
    if args.detector_backend == "coco":
        detector, preprocess = init_coco_detector(device=device)

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    images = sorted(list(Path(args.input_dir).rglob("*.png")) + list(Path(args.input_dir).rglob("*.jpg")))
    out_depth = os.path.join(args.output_dir, "depth")
    out_vis = os.path.join(args.output_dir, "depth_vis")
    out_global = os.path.join(args.output_dir, "depth_global")
    out_mask = os.path.join(args.output_dir, "roi_mask")
    for d in (out_depth, out_vis, out_global, out_mask):
        os.makedirs(d, exist_ok=True)

    for image_path in tqdm(images, desc="ROI fusion inference"):
        rgb_np = np.array(Image.open(image_path).convert("RGB"))
        h, w = rgb_np.shape[:2]

        global_depth = run_depth(pipe, rgb_np, args, generator=generator)

        roi_mask = load_mask_for_image(str(image_path), args.roi_mask_dir)
        if roi_mask is None and detector is not None:
            roi_mask = make_roi_mask_by_detector(
                rgb_np=rgb_np,
                detector=detector,
                preprocess=preprocess,
                labels_keep=labels_keep,
                score_thr=args.detector_score_thr,
            )
        if roi_mask is None:
            roi_mask = np.zeros((h, w), dtype=np.float32)

        boxes = connected_component_boxes(roi_mask, threshold=0.5, min_area=args.roi_min_area)
        crop_preds = []
        for box in boxes:
            x1, y1, x2, y2 = expand_box(box, (h, w), expand_ratio=args.roi_expand_ratio)
            crop = rgb_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_depth = run_depth(pipe, crop, args, generator=generator)
            crop_preds.append(((x1, y1, x2, y2), crop_depth))

        fused_depth, effective_mask = fuse_roi_depth(
            global_depth=global_depth,
            crop_preds=crop_preds,
            roi_mask=roi_mask,
            align_mode=args.align_mode,
            blend_blur_ksize=args.blend_blur_ksize,
            fusion_weight=args.fusion_weight,
        )

        stem = image_path.stem
        np.save(os.path.join(out_depth, f"{stem}.npy"), fused_depth)
        np.save(os.path.join(out_global, f"{stem}.npy"), global_depth)
        Image.fromarray((np.clip(effective_mask, 0.0, 1.0) * 255).astype(np.uint8)).save(
            os.path.join(out_mask, f"{stem}.png")
        )
        colorize_depth_map(global_depth, reverse_color=args.disparity).save(os.path.join(out_global, f"{stem}.png"))
        colorize_depth_map(fused_depth, reverse_color=args.disparity).save(os.path.join(out_vis, f"{stem}.png"))

    logging.info("ROI fusion inference complete. Results: %s", args.output_dir)


if __name__ == "__main__":
    main()
