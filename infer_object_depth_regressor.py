#!/usr/bin/env python
"""Single-image inference: YOLO + regressor object attention + Lotus-D."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import LotusDPipeline
from utils.object_detection_cache import load_detections, load_yolo_model, run_yolo_detections
from utils.object_depth_regressor import ObjectDepthRegressorBundle
from utils.object_attention_condition import padded_object_condition_from_detections
from utils.seed_all import seed_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb", type=str, required=True)
    p.add_argument("--detail_model", type=str, required=True)
    p.add_argument("--regressor_dir", type=str, required=True)
    p.add_argument("--detail_root", type=str, default=None, help="Optional cached detections dir")
    p.add_argument("--detection_score_thr", type=float, default=None)
    p.add_argument("--max_objects", type=int, default=16)
    p.add_argument("--processing_res", type=int, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    rgb_np = np.array(Image.open(args.rgb).convert("RGB"))
    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    regressor = ObjectDepthRegressorBundle.load(args.regressor_dir, device=device)
    detection_score_thr = (
        float(args.detection_score_thr)
        if args.detection_score_thr is not None
        else float(regressor.config.get("detection_score_thr", 0.5))
    )
    processing_res = (
        args.processing_res
        if args.processing_res is not None
        else regressor.config.get("processing_res")
    )

    if args.detail_root:
        detections = [
            d
            for d in load_detections(args.rgb, args.detail_root)
            if d.score >= detection_score_thr
        ]
    else:
        model = load_yolo_model("yolov8n-seg.pt")
        detections = run_yolo_detections(
            rgb_np, model=model, score_thr=detection_score_thr
        )

    detail_pipe = LotusDPipeline.from_pretrained(args.detail_model, torch_dtype=dtype).to(device)
    if int(detail_pipe.unet.config.in_channels) != 4:
        raise ValueError(
            "Object attention inference requires a 4ch detail model; "
            f"got {detail_pipe.unet.config.in_channels}."
        )
    if detail_pipe.object_condition_encoder is None:
        raise ValueError("detail_model has no object_condition_encoder")
    class_ids, object_features, object_mask = (
        padded_object_condition_from_detections(
            detections,
            rgb_np.shape[1],
            rgb_np.shape[0],
            regressor,
            args.max_objects,
            rgb_np=rgb_np,
        )
    )

    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    image = image.to(device)
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    autocast_ctx = (
        nullcontext()
        if device.type in ("cpu", "mps")
        else torch.autocast(device_type=device.type, enabled=args.half_precision)
    )
    with torch.no_grad(), autocast_ctx:
        pred = detail_pipe(
            rgb_in=image,
            prompt="",
            num_inference_steps=1,
            generator=generator,
            output_type="np",
            timesteps=[999],
            task_emb=task_emb,
            object_class_ids=class_ids,
            object_features=object_features,
            object_mask=object_mask,
            processing_res=processing_res,
            match_input_res=True,
        ).images[0]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_disp.npy", pred.mean(axis=-1))
    np.save(out / "object_features.npy", object_features.numpy())
    np.save(out / "object_mask.npy", object_mask.numpy())
    Image.fromarray(rgb_np).save(out / "rgb.png")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
