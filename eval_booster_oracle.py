#!/usr/bin/env python
"""Oracle ceiling on Booster: how much could any material-targeted method recover?

Same question Phase 0 asked on NYUv2 (eval_object_oracle_ceiling.py), asked where it
can actually be answered. NYUv2's Kinect leaves holes on exactly the surfaces of
interest and holes are excluded from scoring; Booster's space-time-stereo GT is dense
on specular and transparent surfaces, and its material masks say which pixels those
are.

Variants, all applied post-hoc to the aligned Lotus prediction:

  A_*   per-region depth LEVEL corrected to GT, predicted shape kept. Ceiling for any
        method that infers "this material sits at distance d" - the material analogue
        of the object-depth regressor that Phase 0 ruled out at 2.1%.
  B_*   region replaced by GT outright. Ceiling for ANY method targeting it.
  easy  the complement (lambertian + class1) replaced, to partition total error.
  *_shift  CONTROL: the same mask circularly translated to an unrelated location.
        Identical area and shape, no material correspondence. Per-scene inspection
        showed some scenes (notably Door) where Lotus gets the whole scene's slant
        wrong; substituting GT anywhere in such a scene helps regardless of material,
        which would inflate the oracle. The material gain is only real to the extent
        it exceeds this control.

Regions are connected components within a class, so each physical object gets its own
correction rather than one global offset for every mirror in the scene.

Alignment happens before substitution; doing it after would let GT leak into the fit.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from evaluation.util.alignment import align_depth_least_square
from evaluation.util.metric import abs_relative_difference, delta1_acc
from eval_booster_mono import CLASS_NAMES, downsample, list_samples

Image.MAX_IMAGE_PIXELS = None

HARD = (2, 3)  # specular + transparent/mirror
EASY = (0, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Material-class oracle ceiling on Booster.")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument(
        "--processing_res",
        type=int,
        default=512,
        help="Which cached prediction to use. 512 is Booster's measured optimum.",
    )
    p.add_argument("--output_dir", type=str, default="output/eval_booster_oracle")
    p.add_argument("--illum", type=str, default="im0")
    p.add_argument("--eval_scale", type=int, default=2)
    p.add_argument("--min_region_px", type=int, default=200)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--seed", type=int, default=42, help="Seed for the shifted-mask control.")
    p.add_argument("--n_controls", type=int, default=3,
                   help="Random offsets averaged per control, to damp its variance.")
    return p.parse_args()


def components(mask: np.ndarray):
    """Connected components of a boolean mask, largest-first."""
    import cv2

    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [(lab == i) for i in range(1, n)]


def oracle_level(depth, gt, region, valid, min_px):
    """Fix each region's depth level, keep its predicted shape (multiplicative)."""
    out = depth.copy()
    for comp in components(region):
        m = comp & valid
        if m.sum() < min_px:
            continue
        pm, gm = float(np.median(depth[m])), float(np.median(gt[m]))
        if pm <= 1e-9 or not np.isfinite(gm):
            continue
        out[m] = depth[m] * (gm / pm)
    return out


def oracle_replace(depth, gt, region, valid):
    out = depth.copy()
    m = region & valid
    out[m] = gt[m]
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    root = Path(args.data_root)
    cache = Path(args.pred_cache_dir) / f"res{args.processing_res}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = list_samples(root, args.illum)
    if args.max_scenes > 0:
        keep = set(sorted({s.name for s, _, _ in samples})[: args.max_scenes])
        samples = [t for t in samples if t[0].name in keep]

    # Each real mask is paired with a shifted control of identical area and shape.
    # Controls are averaged over several random offsets to damp their variance.
    REGIONS = ["c2", "c3", "hard"]
    variants = ["baseline", "B_easy"]
    for r in REGIONS:
        variants += [f"A_{r}", f"B_{r}", f"A_{r}_ctrl", f"B_{r}_ctrl"]
    rng = np.random.default_rng(args.seed)
    acc = {v: {"err": 0.0, "d1": 0.0, "n": 0, "touched": 0} for v in variants}
    share = {"c2": 0, "c3": 0, "hard": 0, "easy": 0, "all": 0}
    rows = []

    for scene, frame, _ in tqdm(samples, desc="booster_oracle"):
        cpath = cache / scene.name / f"{frame}_pred.npy"
        if not cpath.is_file():
            raise FileNotFoundError(
                f"Missing cached prediction: {cpath}\n"
                f"Run: python eval_booster_mono.py --processing_res {args.processing_res} ..."
            )
        k = args.eval_scale
        pred = downsample(np.load(cpath).astype(np.float32), k)
        gt_disp = downsample(np.load(scene / "disp_00.npy").astype(np.float32), k)
        occ = downsample(np.array(Image.open(scene / "mask_00.png")), k, nearest=True)
        cat = downsample(np.array(Image.open(scene / "mask_cat.png")), k, nearest=True)

        valid = np.isfinite(gt_disp) & (gt_disp > 0) & (occ > 127) & np.isfinite(pred) & (pred > 0)
        if valid.sum() < 1000:
            continue

        aligned = align_depth_least_square(
            gt_arr=gt_disp.astype(np.float64),
            pred_arr=pred.astype(np.float64),
            valid_mask_arr=valid,
            return_scale_shift=False,
        )
        base = 1.0 / np.clip(aligned, 1e-3, None)
        gt = 1.0 / np.clip(gt_disp.astype(np.float64), 1e-3, None)

        m_c2, m_c3 = cat == 2, cat == 3
        m_hard = np.isin(cat, HARD)
        m_easy = np.isin(cat, EASY)

        masks = {"c2": m_c2, "c3": m_c3, "hard": m_hard}
        h, w = m_hard.shape
        n = int(valid.sum())
        vt, gt_t = torch.from_numpy(valid), torch.from_numpy(gt)

        def score(d):
            return (
                float(abs_relative_difference(torch.from_numpy(d), gt_t, vt)),
                float(delta1_acc(torch.from_numpy(d), gt_t, vt)),
            )

        row = {"scene": scene.name, "frame": frame}

        def record(name, a, d1, touched):
            acc[name]["err"] += a * n
            acc[name]["d1"] += d1 * n
            acc[name]["n"] += n
            acc[name]["touched"] += touched
            row[f"absrel_{name}"] = a

        record("baseline", *score(base), 0)
        record("B_easy", *score(oracle_replace(base, gt, m_easy, valid)),
               int((m_easy & valid).sum()))

        for key, m in masks.items():
            record(f"A_{key}", *score(oracle_level(base, gt, m, valid, args.min_region_px)),
                   int((m & valid).sum()))
            record(f"B_{key}", *score(oracle_replace(base, gt, m, valid)),
                   int((m & valid).sum()))

            # controls: same shape and area, unrelated location, averaged over offsets
            ca = cd = cb = cbd = 0.0
            ctouch = 0
            for _ in range(args.n_controls):
                sh = np.roll(m, (int(rng.integers(h // 5, 4 * h // 5)),
                                 int(rng.integers(w // 5, 4 * w // 5))), axis=(0, 1))
                a1, d1_ = score(oracle_level(base, gt, sh, valid, args.min_region_px))
                a2, d2_ = score(oracle_replace(base, gt, sh, valid))
                ca += a1; cd += d1_; cb += a2; cbd += d2_
                ctouch += int((sh & valid).sum())
            k_ = args.n_controls
            record(f"A_{key}_ctrl", ca / k_, cd / k_, ctouch // k_)
            record(f"B_{key}_ctrl", cb / k_, cbd / k_, ctouch // k_)
        share["c2"] += int((m_c2 & valid).sum())
        share["c3"] += int((m_c3 & valid).sum())
        share["hard"] += int((m_hard & valid).sum())
        share["easy"] += int((m_easy & valid).sum())
        share["all"] += n
        rows.append(row)

    if not rows:
        raise RuntimeError("No samples scored.")

    base_absrel = acc["baseline"]["err"] / acc["baseline"]["n"]
    summary = {
        "num_samples": len(rows),
        "processing_res": args.processing_res,
        "eval_scale": args.eval_scale,
        "pixel_share": {k: share[k] / share["all"] for k in ("c2", "c3", "hard", "easy")},
        "variants": {},
    }
    for v in variants:
        a = acc[v]["err"] / acc[v]["n"]
        summary["variants"][v] = {
            "abs_rel": a,
            "delta1": acc[v]["d1"] / acc[v]["n"],
            "gain": base_absrel - a,
            "gain_pct": 100.0 * (base_absrel - a) / base_absrel if base_absrel else 0.0,
            "touched_share": acc[v]["touched"] / max(acc["baseline"]["n"], 1),
        }
    # A material effect only counts insofar as it beats its own shifted control.
    summary["control_corrected"] = {}
    for r in ["c2", "c3", "hard"]:
        for k in ("A", "B"):
            real = summary["variants"][f"{k}_{r}"]["gain_pct"]
            ctrl = summary["variants"][f"{k}_{r}_ctrl"]["gain_pct"]
            summary["control_corrected"][f"{k}_{r}"] = {
                "real": real, "control": ctrl, "net": real - ctrl,
                "touched_real": summary["variants"][f"{k}_{r}"]["touched_share"],
                "touched_ctrl": summary["variants"][f"{k}_{r}_ctrl"]["touched_share"],
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        cols = sorted({k for r in rows for k in r})
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    ps = summary["pixel_share"]
    print(f"\nBooster oracle  res={args.processing_res}  n={len(rows)}  controls/mask={args.n_controls}")
    print(f"pixel share: c2={ps['c2']*100:.1f}%  c3={ps['c3']*100:.1f}%  hard={ps['hard']*100:.1f}%")
    print(f"baseline abs_rel = {base_absrel:.5f}\n")
    print(f"{'region':<8}{'variant':<10}{'real':>9}{'control':>10}{'NET':>9}{'px real':>10}{'px ctrl':>9}")
    print("-" * 66)
    for r in ["c2", "c3", "hard"]:
        for k in ("A", "B"):
            c = summary["control_corrected"][f"{k}_{r}"]
            tag = "level" if k == "A" else "full GT"
            print(f"{r:<8}{tag:<10}{c['real']:>8.1f}%{c['control']:>9.1f}%{c['net']:>8.1f}%"
                  f"{c['touched_real']*100:>9.1f}%{c['touched_ctrl']*100:>8.1f}%")
    print(f"\nB c0+c1 full GT: {summary['variants']['B_easy']['gain_pct']:.1f}%")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
