#!/usr/bin/env python
"""Does telling Lotus what objects are in the image help it find invisible depth steps?

docs/rgb_edge_route_closure.md and the edge-dependence run located the real deficit:
Lotus recovers 31.2% of true depth discontinuities that sit on an RGB edge but only 10.5%
of those in visually flat regions - and the flat ones are 54.6% of all of them. A chair
the same colour as the wall behind it has a depth step with nothing in the image to mark
it. A human still knows the step is there, because they know it is a chair.

Lotus descends from Stable Diffusion and still carries the CLIP text encoder; the pipeline
feeds prompt embeddings to the UNet as encoder_hidden_states and is currently called with
prompt="". So the channel for that knowledge exists and is sitting unused, and testing it
costs inference only - no architecture change, no training.

Two reasons for caution, which is why the control matters more than the treatment here.
Class names carry no location, and the deficit is a localisation deficit. And Lotus was
fine-tuned with empty prompts, so its cross-attention is adapted to empty text; real text
may act as perturbation rather than information.

  empty      prompt="" - what every other experiment in this repo used
  classes    the YOLO class names found in THIS image
  shuffled   CONTROL: class names from a DIFFERENT image. Same kind of text, same length
             distribution, wrong content. Any gain that survives this is semantic; a gain
             that does not is the text channel being jostled.
  generic    "an indoor scene" - fixed text, tests whether merely leaving empty matters
  classes_pos       names WITH a coarse location: "a chair on the left, a sink at the
                    bottom right". A name alone cannot localise anything, and the deficit
                    is a localisation deficit, so this is the version of the idea that
                    could in principle address it.
  classes_wrongpos  CONTROL: right names, positions randomly reassigned. Isolates the
                    position words from the names they are attached to.

Scored on abs_rel and delta1, on boundary F1, and on the split that motivated the run:
recall of true discontinuities on and off RGB edges.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_boundary_f1 import boundary_f1
from eval_edge_dependence import within
from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path, align_to_gt, score
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs
from pipeline import LotusDPipeline
from utils.object_detection_cache import load_detections

VARIANTS = ["empty", "classes", "shuffled", "generic", "classes_pos", "classes_wrongpos"]


def parse_args():
    p = argparse.ArgumentParser(description="Class-name text conditioning for Lotus.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--detail_artifacts_dir", type=str, default="D:/lotus/data/nyuv2_detail_artifacts/test")
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/prompt_cache")
    p.add_argument("--output_dir", type=str, default="output/eval_text_prompt")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--generic_prompt", type=str, default="an indoor scene")
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--canny", type=int, nargs=2, default=(50, 150))
    p.add_argument("--half_precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def predict(pipe, rgb_np, prompt, timestep, processing_res, generator):
    device = pipe.device
    image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    image = (image / 127.5 - 1.0).to(device)
    task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
    ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device_type=device.type)
    with ctx:
        out = pipe(
            rgb_in=image, prompt=prompt, num_inference_steps=1, generator=generator,
            output_type="np", timesteps=[timestep], task_emb=task_emb,
            processing_res=processing_res, match_input_res=True,
        ).images[0]
    return (out.mean(axis=-1) if out.ndim == 3 else out).astype(np.float32)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_artifacts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]
    rng0 = np.random.default_rng(args.seed)

    # class names per image, then a derangement of them for the control
    cells = [["top left", "top centre", "top right"],
             ["centre left", "the centre", "centre right"],
             ["bottom left", "bottom centre", "bottom right"]]

    def cell_of(det, h, w):
        x1, y1, x2, y2 = det.bbox
        return (min(int((y1 + y2) / 2 / max(h, 1) * 3), 2),
                min(int((x1 + x2) / 2 / max(w, 1) * 3), 2))

    def phrase(det, h, w):
        r, c = cell_of(det, h, w)
        return f"a {det.class_name} at {cells[r][c]}"

    names, named_pos, named_wrongpos = [], [], []
    for rgb_path, _ in pairs:
        dets = [d for d in load_detections(rgb_path, detail_root) if d.score >= args.detection_score_thr]
        uniq = sorted({d.class_name for d in dets})
        names.append(", ".join(uniq))
        ih, iw = Image.open(rgb_path).size[1], Image.open(rgb_path).size[0]
        named_pos.append(", ".join(phrase(d, ih, iw) for d in dets))
        # Same names, each given a cell that is NOT its own. Permuting cells among the
        # image's own detections was the first attempt and was degenerate: 58% of images
        # hold one detection or none, and a third of the rest put every detection in the
        # same cell, so 73% of controls came out as the identical string.
        if dets:
            wrong = []
            for d in dets:
                r0, c0 = cell_of(d, ih, iw)
                while True:
                    r, c = int(rng0.integers(3)), int(rng0.integers(3))
                    if (r, c) != (r0, c0):
                        break
                wrong.append(f"a {d.class_name} at {cells[r][c]}")
            named_wrongpos.append(", ".join(wrong))
        else:
            named_wrongpos.append("")
    order = rng0.permutation(len(names))
    for i in range(len(order)):  # avoid an image being given its own names
        if order[i] == i:
            j = (i + 1) % len(order)
            order[i], order[j] = order[j], order[i]
    shuffled = [names[o] for o in order]
    n_named = sum(1 for s in names if s)
    logging.info("Images: %d  with class names: %d", len(pairs), n_named)
    logging.info("  names   : %r", next((s for s in names if s), ""))
    logging.info("  with pos: %r", next((s for s in named_pos if s), ""))

    dtype = torch.float16 if args.half_precision else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    thresholds = np.linspace(5.0, 25.0, 11)
    weights = thresholds / thresholds.sum()
    acc = {v: {"absrel": [], "d1": [], "bf1": [], "on": [0, 0], "off": [0, 0]} for v in VARIANTS}

    for idx, (rgb_path, depth_path) in enumerate(tqdm(pairs, desc="prompts")):
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)
        if valid.sum() < 100:
            continue
        rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
        grey = np.array(Image.open(rgb_path).convert("L"))
        edge = within(cv2.Canny(grey, *args.canny).astype(bool) & valid, 1)
        gt_d = discontinuities(gt, valid, args.t)
        gt_on, gt_off = gt_d & edge, gt_d & ~edge

        prompts = {"empty": "", "classes": names[idx],
                   "shuffled": shuffled[idx], "generic": args.generic_prompt,
                   "classes_pos": named_pos[idx], "classes_wrongpos": named_wrongpos[idx]}
        for v, prompt in prompts.items():
            cache = Path(args.pred_cache_dir) / f"res{args.processing_res}" / v
            cp = _cache_path(rgb_path, rgb_dir, cache, "_pred.npy")
            if cp.is_file():
                pred = np.load(cp).astype(np.float64)
            else:
                g = torch.Generator(device=device).manual_seed(args.seed)
                pred = predict(pipe, rgb_np, prompt, args.timestep, args.processing_res, g)
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.save(cp, pred.astype(np.float16))
                pred = pred.astype(np.float64)
            base = align_to_gt(pred, gt, valid)
            if base is None:
                continue
            a, d1 = score(base, gt, valid)
            acc[v]["absrel"].append(a)
            acc[v]["d1"].append(d1)
            # boundary_f1 returns nan for a threshold no contour reaches; renormalise the
            # weights over the thresholds that produced a value instead of poisoning the mean
            c = boundary_f1(base, gt, valid, thresholds)
            ok = np.isfinite(c)
            acc[v]["bf1"].append(float((c[ok] * weights[ok]).sum() / weights[ok].sum())
                                 if ok.any() else np.nan)
            lo_d = within(discontinuities(base, valid, args.t), 1)
            acc[v]["on"][0] += int((gt_on & lo_d).sum())
            acc[v]["on"][1] += int(gt_on.sum())
            acc[v]["off"][0] += int((gt_off & lo_d).sum())
            acc[v]["off"][1] += int(gt_off.sum())

    summary = {"n_images": len(acc["empty"]["absrel"]), "images_with_names": n_named,
               "generic_prompt": args.generic_prompt, "variants": {}}
    for v in VARIANTS:
        e = acc[v]
        summary["variants"][v] = {
            "abs_rel": float(np.mean(e["absrel"])), "delta1": float(np.mean(e["d1"])),
            "bf1": float(np.nanmean(e["bf1"])),
            "recall_on_edge": e["on"][0] / max(e["on"][1], 1),
            "recall_off_edge": e["off"][0] / max(e["off"][1], 1),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    V = summary["variants"]
    b = V["empty"]
    print(f"\nText-prompt conditioning  n={summary['n_images']}  "
          f"({n_named} images have class names)")
    print(f"\n{'prompt':<10}{'abs_rel':>10}{'vs empty':>10}{'delta1':>9}"
          f"{'BF1':>9}{'recall ON':>11}{'recall OFF':>12}")
    print("-" * 71)
    for v in VARIANTS:
        d = V[v]
        da = 100.0 * (b["abs_rel"] - d["abs_rel"]) / b["abs_rel"]
        print(f"{v:<10}{d['abs_rel']:>10.5f}{da:>9.2f}%{d['delta1']:>9.4f}{d['bf1']:>9.4f}"
              f"{d['recall_on_edge']*100:>10.1f}%{d['recall_off_edge']*100:>11.1f}%")
    print("\n'recall OFF' is the deficit this tests: true depth steps with no image edge.")
    print("classes must beat shuffled, not just empty, for the effect to be semantic.")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
