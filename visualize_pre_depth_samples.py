#!/usr/bin/env python
"""Render RGB, regressor pre-depth, valid mask, and ROI overlay samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.object_pre_depth import load_pre_depth_artifacts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pre_depth_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    return parser.parse_args()


def render_sample(rgb_path: Path, pre_depth: np.ndarray, valid: np.ndarray, output: Path):
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    h, w = rgb.shape[:2]
    pre = np.asarray(
        Image.fromarray(pre_depth.astype(np.float32)).resize(
            (w, h), Image.Resampling.BILINEAR
        )
    )
    mask = np.asarray(
        Image.fromarray(valid.astype(np.uint8)).resize(
            (w, h), Image.Resampling.NEAREST
        )
    ) > 0
    pre_01 = np.clip((pre + 1.0) * 0.5, 0.0, 1.0)
    color = plt.get_cmap("turbo")(pre_01)[..., :3]
    overlay = rgb.astype(np.float32) / 255.0
    overlay[mask] = 0.35 * overlay[mask] + 0.65 * color[mask]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(pre_01, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title("Calibrated pre-depth")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Valid object mask")
    axes[3].imshow(overlay)
    axes[3].set_title("Object-region overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(rgb_path.name)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for entry in payload["entries"]:
        rgb_path = Path(entry["rgb_path"])
        pre_depth, valid = load_pre_depth_artifacts(rgb_path, args.pre_depth_root)
        if pre_depth is None or valid is None or not np.any(valid):
            continue
        output = args.output_dir / f"{len(rendered):02d}_{rgb_path.stem}.png"
        render_sample(rgb_path, pre_depth, valid, output)
        rendered.append(output)
        if len(rendered) >= args.num_samples:
            break
    if not rendered:
        raise FileNotFoundError("No pre-depth samples with a valid object mask were found.")

    images = [Image.open(path).convert("RGB") for path in rendered]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet_path = args.output_dir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(sheet_path)


if __name__ == "__main__":
    main()
