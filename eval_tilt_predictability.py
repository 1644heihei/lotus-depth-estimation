#!/usr/bin/env python
"""Can the global tilt error be predicted without ground truth?

Removing a 3-parameter tilt from the residual recovers 22.8% on Booster, but that fit
uses GT, so it is an oracle rather than a method. The tilt is also per-image
(|mean|/sigma 0.11, sign agreement 58%), so no fixed correction reaches it, and it
survives averaging because every perturbation makes the same structural mistake
(global residual correlation 0.67-0.92).

That leaves one question, and it is empirical rather than rhetorical: is the tilt
predictable from signals available at inference time? Whatever a learned corrector could
recover is bounded by that predictability.

  R^2 near zero -> the information is not in the outputs; only extra input (intrinsics,
                   a second view, sparse depth) or a better model reaches the 22.8%
  R^2 high      -> a small regressor recovers a real share of it

Features are all GT-free: statistics of the prediction itself, of the RGB, and of the
disagreement between TTA members. Cross-validation is grouped BY SCENE, since frames of
one scene differ only in illumination and would otherwise leak the answer across folds.

The headline is not R^2 but the abs_rel gain from applying the cross-validated
correction, which is directly comparable to the 22.8% oracle.
"""

from __future__ import annotations

import argparse
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
from eval_booster_mono import downsample, list_samples
from eval_booster_multires_ensemble import fit_to_reference

Image.MAX_IMAGE_PIXELS = None


def parse_args():
    p = argparse.ArgumentParser(description="Is the global tilt predictable without GT?")
    p.add_argument("--data_root", type=str, default="D:/lotus/data/booster/extracted/train/balanced")
    p.add_argument("--pred_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache")
    p.add_argument("--flip_cache_dir", type=str, default="D:/lotus/data/booster/pred_cache_flip")
    p.add_argument("--resolutions", type=int, nargs="+", default=[384, 512, 640, 768])
    p.add_argument("--reference", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="output/eval_tilt_predictability")
    p.add_argument("--illum", type=str, default="all")
    p.add_argument("--eval_scale", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def plane_basis(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx / w - 0.5).astype(np.float64), (yy / h - 0.5).astype(np.float64)


def fit_tilt(field, valid, x, y):
    """Least-squares a, b, c for a*x + b*y + c over valid pixels."""
    A = np.stack([x[valid], y[valid], np.ones(valid.sum())], axis=1)
    coef, *_ = np.linalg.lstsq(A, field[valid], rcond=None)
    return coef  # a, b, c


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = list(args.resolutions)

    feats, targets, groups, keep = [], [], [], []

    for scene, frame, rgb_path in tqdm(list_samples(root, args.illum), desc="tilt_features"):
        k = args.eval_scale
        P = {}
        ok = True
        for r in res:
            f = Path(args.pred_cache_dir) / f"res{r}" / scene.name / f"{frame}_pred.npy"
            if not f.is_file():
                ok = False
                break
            P[r] = downsample(np.load(f).astype(np.float32), k)
        if not ok:
            continue
        fp = Path(args.flip_cache_dir) / f"res{args.reference}" / scene.name / f"{frame}_pred.npy"
        flip = downsample(np.load(fp).astype(np.float32), k) if fp.is_file() else None

        gt = downsample(np.load(scene / "disp_00.npy").astype(np.float32), k)
        occ = downsample(np.array(Image.open(scene / "mask_00.png")), k, nearest=True)
        valid = np.isfinite(gt) & (gt > 0) & (occ > 127)
        for r in res:
            valid &= np.isfinite(P[r]) & (P[r] > 0)
        if valid.sum() < 500:
            continue

        ref = P[args.reference]
        h, w = ref.shape
        x, y = plane_basis(h, w)
        gtd = gt.astype(np.float64)

        aligned = align_depth_least_square(
            gt_arr=gtd, pred_arr=ref.astype(np.float64), valid_mask_arr=valid, return_scale_shift=False
        )
        rng = float(np.percentile(gtd[valid], 99) - np.percentile(gtd[valid], 1))
        if rng <= 1e-9:
            continue
        # TARGET: the tilt of the residual against GT, normalised by scene disparity range
        ta, tb, _ = fit_tilt((aligned - gtd) / rng, valid, x, y)

        # ---- GT-free features -------------------------------------------------
        rp = ref.astype(np.float64)
        prng = float(np.percentile(rp[valid], 99) - np.percentile(rp[valid], 1)) or 1.0
        pa, pb, _ = fit_tilt(rp / prng, valid, x, y)          # the prediction's own tilt
        rgb = np.array(Image.open(rgb_path).convert("L").resize((w, h))).astype(np.float64) / 255.0
        ia, ib, _ = fit_tilt(rgb, valid, x, y)                 # image brightness gradient

        members = [fit_to_reference(P[r], ref, valid) for r in res]
        if flip is not None:
            members.append(fit_to_reference(flip, ref, valid))
        S = np.stack(members)
        spread = S.std(0) / prng
        sa, sb, sc_ = fit_tilt(spread, valid, x, y)            # tilt of the disagreement
        # difference between the extreme members carries the direction they disagree in
        d_ext = (S[0] - S[-1]) / prng
        da, db, _ = fit_tilt(d_ext, valid, x, y)

        vv = rp[valid] / prng
        feats.append([
            pa, pb, ia, ib, sa, sb, sc_, da, db,
            float(spread[valid].mean()), float(spread[valid].std()),
            float(vv.mean()), float(vv.std()),
            float(np.percentile(vv, 90) - np.percentile(vv, 10)),
            float(rgb[valid].mean()), float(rgb[valid].std()),
            float(valid.mean()),
        ])
        targets.append([ta, tb])
        groups.append(scene.name)
        keep.append((scene.name, frame))

    X = np.array(feats)
    Y = np.array(targets)
    G = np.array(groups)
    logging.info("samples=%d  scenes=%d  features=%d", len(X), len(set(G)), X.shape[1])

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=min(8, len(set(G))))
    pred = np.zeros_like(Y)
    for tr, te in gkf.split(X, Y, groups=G):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        m.fit(X[tr], Y[tr])
        pred[te] = m.predict(X[te])

    def r2(y, p):
        ss = ((y - p) ** 2).sum()
        return 1.0 - ss / max(((y - y.mean()) ** 2).sum(), 1e-12)

    r2a, r2b = r2(Y[:, 0], pred[:, 0]), r2(Y[:, 1], pred[:, 1])
    summary = {
        "num_samples": int(len(X)),
        "num_scenes": int(len(set(G))),
        "r2_tilt_x": float(r2a),
        "r2_tilt_y": float(r2b),
        "corr_x": float(np.corrcoef(Y[:, 0], pred[:, 0])[0, 1]),
        "corr_y": float(np.corrcoef(Y[:, 1], pred[:, 1])[0, 1]),
    }

    # ---- apply the cross-validated correction and measure the real gain ----
    idx = {kf: i for i, kf in enumerate(keep)}
    acc = {"baseline": [0.0, 0.0, 0], "cv_corrected": [0.0, 0.0, 0], "oracle_tilt": [0.0, 0.0, 0]}
    for scene, frame, _ in tqdm(list_samples(root, args.illum), desc="apply"):
        i = idx.get((scene.name, frame))
        if i is None:
            continue
        k = args.eval_scale
        ref = downsample(np.load(Path(args.pred_cache_dir) / f"res{args.reference}" /
                                scene.name / f"{frame}_pred.npy").astype(np.float32), k)
        gt = downsample(np.load(scene / "disp_00.npy").astype(np.float32), k)
        occ = downsample(np.array(Image.open(scene / "mask_00.png")), k, nearest=True)
        valid = np.isfinite(gt) & (gt > 0) & (occ > 127) & np.isfinite(ref) & (ref > 0)
        if valid.sum() < 500:
            continue
        h, w = ref.shape
        x, y = plane_basis(h, w)
        gtd = gt.astype(np.float64)
        al = align_depth_least_square(
            gt_arr=gtd, pred_arr=ref.astype(np.float64), valid_mask_arr=valid, return_scale_shift=False
        )
        rng = float(np.percentile(gtd[valid], 99) - np.percentile(gtd[valid], 1)) or 1.0
        ta, tb, _ = fit_tilt((al - gtd) / rng, valid, x, y)
        pa, pb = pred[i]

        n = int(valid.sum())
        vt = torch.from_numpy(valid)
        gtt = torch.from_numpy(1.0 / np.clip(gtd, 1e-3, None))
        floor = max(1e-3, 0.02 * float(np.median(gtd[valid])))
        for name, corr in [("baseline", None), ("cv_corrected", (pa, pb)), ("oracle_tilt", (ta, tb))]:
            d = al if corr is None else al - rng * (corr[0] * x + corr[1] * y)
            dd = torch.from_numpy(1.0 / np.clip(d, floor, None))
            e = acc[name]
            e[0] += float(abs_relative_difference(dd, gtt, vt)) * n
            e[1] += float(delta1_acc(dd, gtt, vt)) * n
            e[2] += n

    base = acc["baseline"][0] / acc["baseline"][2]
    for kname in acc:
        a = acc[kname][0] / acc[kname][2]
        summary[kname] = {"abs_rel": a, "delta1": acc[kname][1] / acc[kname][2],
                          "gain_pct": 100.0 * (base - a) / base}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nTilt predictability  n={summary['num_samples']}  scenes={summary['num_scenes']}"
          f"  (GroupKFold by scene)")
    print(f"  R^2  x-tilt {r2a:+.3f}   y-tilt {r2b:+.3f}")
    print(f"  corr x-tilt {summary['corr_x']:+.3f}   y-tilt {summary['corr_y']:+.3f}")
    print(f"\n{'variant':<16}{'abs_rel':>10}{'delta1':>9}{'gain':>9}")
    print("-" * 46)
    for kname in ["baseline", "cv_corrected", "oracle_tilt"]:
        s = summary[kname]
        print(f"{kname:<16}{s['abs_rel']:>10.5f}{s['delta1']:>9.4f}{s['gain_pct']:>8.2f}%")
    print("\ncv_corrected is what a learned post-hoc corrector could actually deliver;")
    print("oracle_tilt is the ceiling that same correction reaches when it may look at GT.")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
