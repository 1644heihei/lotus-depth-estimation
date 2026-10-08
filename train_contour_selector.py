#!/usr/bin/env python
"""Learn which SAM contour pixels sit on a real depth discontinuity.

docs/contour_selector_plan.md establishes what this has to beat and why. SAM finds the
boundaries - a true depth step is a median 1.00px from the nearest contour - but 79.8% of
the contour pixels it draws sit where no step exists, and the barrier-sharpening operation
spends itself flattening smooth surface there. Keeping only the right 4% with GT lifts the
raw BF1 effect from -1.90% to +43.39% without moving a single contour.

The target is stated in precision, not AUC, because §2.5 measured the conversion:

    raw BF1 gain = 0.69 * (precision at matched retention - 37.1%)

so break-even is 37.1%, the plan's +10% needs 51.6%, and a zero-training threshold on the
best hand-crafted feature already reaches 34.6%. A model that does not clear 34.6% has
bought nothing; one that does not clear 37.1% still loses BF1.

The prediction is affine-invariant log depth, so its level and scale carry no information
and its gradients carry an unknown per-image factor. Inputs are standardised per image, or
training and inference see different magnitudes. That step and the rest of the input
pipeline live in `contour_selector.inputs`, which apply_contour_selector.py reads through
too: a disagreement between the two is silent, and shows up only as a worse BF1.

A note on a constraint this file used to assert and measurement withdrew. Per-pixel random
selection scores -20.35%, and the plan read that as proof that the selection has to be
spatially coherent, since an isolated pixel is no barrier. It is not: the GT oracle's
selection is just as fragmented (component length median 11.3px, 67.8% under 20px) and
scores +43.39%. The -20.35% is explained by its precision of 7.6% alone. A component-size
floor was tried on that reasoning and measured 3.9 points WORSE (+8.61% against +12.49%),
so none is applied - convolution is smooth enough on its own.

Trains on NYUv2 train (795 frames) by default, validates on a scene-disjoint split of it,
and never touches the 654-frame test set - that is scored separately once, by the
sharpening pipeline. --mix_json trains on several datasets at once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contour_selector import (PRED_SPACES, UNet, build_input, contour_labels,
                              in_channels, load_candidates, load_prediction,
                              top_share_mask_torch)
from utils.eval_frames import (add_dataset_args, depth_scale, list_frames,
                               valid_mask)

NYU = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar"


def parse_args():
    p = argparse.ArgumentParser(description="Train the contour selector.")
    p.add_argument("--rgb_dir", type=str, default=f"{NYU}/train")
    p.add_argument("--pred_cache_dir", type=str,
                   default="D:/lotus/data/marigold_v2_pred_train/res640")
    p.add_argument("--mask_cache_dir", type=str,
                   default="D:/lotus/data/oracle_cache/sam_auto48_train")
    p.add_argument("--output_dir", type=str, default="output/contour_selector")
    p.add_argument("--t", type=float, default=10.0, help="Depth-step threshold in percent.")
    p.add_argument("--label_px", type=float, default=1.0)
    p.add_argument("--val_scenes", type=int, default=40, help="Scenes held out of training.")
    p.add_argument("--val_frac", type=float, default=0.0,
                   help="Hold out this FRACTION of each source's scenes instead of a "
                        "fixed count. Use it with --mix_json: sources differ in how many "
                        "scenes they have, and a fixed 40 would hold out 16%% of NYUv2 "
                        "but most of a smaller source.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--retention", type=float, default=0.06,
                   help="Fraction of contour pixels kept when scoring precision, matched "
                        "to what the end-to-end runs used.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument(
        "--no_rgb",
        action="store_true",
        help="Ablation: drop the three RGB channels. The AUC sweep found every usable "
             "feature to be depth-derived (RGB gradient 0.54, contour density 0.53, both "
             "near chance), so RGB may contribute nothing but a route to memorising a "
             "dataset's appearance - the selector keeps 50.2% precision on NYUv2 and falls "
             "to 17.4% on ScanNet while a hand-crafted threshold holds 25.0%.",
    )
    p.add_argument(
        "--no_sam",
        action="store_true",
        help="Ablation: drop the SAM contour channel AND the restriction to SAM contour "
             "pixels, so the network must find the boundaries itself from RGB and depth. "
             "This is the setting a depth-only refiner works in (Ramamonjisoa et al., CVPR "
             "2020 use no segmentation at all), and it tests whether SAM is load-bearing. "
             "Pair it with a --retention matched in ABSOLUTE pixel count, since candidates "
             "become every valid pixel rather than the ~9% that lie on a SAM contour.",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--pred_space",
        choices=PRED_SPACES,
        default="log",
        help="What the cached predictions hold, for every source that does not "
             "override it. Marigold V2 emits affine-invariant log depth; Lotus "
             "emits disparity, which has to be converted or the selector is fed "
             "an inverted signal. A --mix_json source may carry its own "
             "pred_space, so one selector can be trained across both.",
    )
    p.add_argument(
        "--mix_json",
        type=str,
        default="",
        help="Train on several datasets at once. A JSON list of sources, each with "
             "dataset / rgb_dir / pred_cache_dir / mask_cache_dir and an optional "
             "max_images, replacing the single-source arguments above. The scene-level "
             "split is applied WITHIN each source, so both train and val hold every "
             "domain - selecting the epoch on one domain's val would defeat the point of "
             "mixing. Sources of unequal frame size need --crop.",
    )
    p.add_argument(
        "--crop",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Random crop every frame to H x W. Needed to batch sources of different "
             "size (NYUv2 is 480x640, Hypersim 768x1024). Crop rather than resize: the "
             "sharpening band is defined in PIXELS, so rescaling would change what a "
             "2px band means and the selector would learn the wrong scale. A frame "
             "already at or below the crop size is passed through untouched.",
    )
    add_dataset_args(p)
    return p.parse_args()


def mix_sources(args):
    """The list of sources to train on: either --mix_json, or the single-source args.

    Each source carries its own dataset name because depth scale and valid mask differ
    (utils/eval_frames.py), and its own caches because predictions and SAM masks are
    stored per dataset.
    """
    if not args.mix_json:
        return [{"dataset": args.dataset, "rgb_dir": args.rgb_dir,
                 "pred_cache_dir": args.pred_cache_dir,
                 "mask_cache_dir": args.mask_cache_dir,
                 "max_images": args.max_images,
                 "pred_space": args.pred_space}]
    srcs = json.loads(Path(args.mix_json).read_text(encoding="utf-8"))
    need = ("dataset", "rgb_dir", "pred_cache_dir", "mask_cache_dir")
    for i, s in enumerate(srcs):
        missing = [k for k in need if k not in s]
        assert not missing, f"--mix_json source {i} is missing {missing}"
        s.setdefault("max_images", 0)
        s.setdefault("pred_space", getattr(args, "pred_space", "log"))
        assert s["pred_space"] in PRED_SPACES, (
            f"--mix_json source {i} has pred_space {s['pred_space']!r}; "
            f"expected one of {PRED_SPACES}")
    return srcs


# ---------------------------------------------------------------- data


class ContourDataset(torch.utils.data.Dataset):
    """Five input channels and a label defined only on contour pixels.

    The prediction is standardised per image: Log-stage2 emits affine-invariant log depth,
    so its absolute level and scale mean nothing and would differ between training (where
    a prediction could be aligned to GT) and inference (where it cannot be).
    """

    def __init__(self, pairs, rgb_dir, pred_dir, mask_dir, t, label_px, no_sam=False,
                 no_rgb=False, dataset="nyuv2", crop=None, pred_space="log"):
        self.pairs = list(pairs)
        self.rgb_dir, self.pred_dir, self.mask_dir = Path(rgb_dir), Path(pred_dir), Path(mask_dir)
        self.t, self.label_px = t, label_px
        self.no_sam = bool(no_sam)
        self.no_rgb = bool(no_rgb)
        self.dataset = dataset
        self.pred_space = pred_space
        self.crop = None if crop is None else tuple(int(v) for v in crop)
        self._rng = None

    def _crop_box(self, h, w, candidates):
        """Top-left corner of a crop that holds at least one candidate pixel.

        Retried a few times rather than searched exhaustively: a crop with no contour in
        it contributes nothing to the masked loss, but it is not wrong, so a cheap retry
        is enough and the loop always terminates.
        """
        ch, cw = self.crop
        if h <= ch and w <= cw:
            return None
        ch, cw = min(ch, h), min(cw, w)
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(None if info is None else info.seed % (2 ** 32))
        for _ in range(8):
            y = int(self._rng.integers(0, h - ch + 1))
            x = int(self._rng.integers(0, w - cw + 1))
            if candidates[y:y + ch, x:x + cw].any():
                return y, x, ch, cw
        return y, x, ch, cw

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rgb_path, depth_path = self.pairs[i]
        gt = np.array(Image.open(depth_path)).astype(np.float64) / depth_scale(self.dataset)
        h, w = gt.shape
        valid = valid_mask(self.dataset, gt)

        pred = load_prediction(rgb_path, self.rgb_dir, self.pred_dir, self.pred_space)
        candidates = load_candidates(rgb_path, self.rgb_dir, self.mask_dir, valid,
                                     no_sam=self.no_sam)
        label = contour_labels(gt, valid, self.t, self.label_px)
        x = build_input(rgb_path, pred, candidates,
                        no_rgb=self.no_rgb, no_sam=self.no_sam)
        if self.crop is not None:
            # After the distance transform and the per-image standardisation, both of
            # which must see the whole frame: at inference the selector gets a full
            # image, and a label computed inside the crop would miss contours just
            # outside it.
            box = self._crop_box(h, w, candidates)
            if box is not None:
                y0, x0, ch, cw = box
                x = x[:, y0:y0 + ch, x0:x0 + cw]
                label = label[y0:y0 + ch, x0:x0 + cw]
                candidates = candidates[y0:y0 + ch, x0:x0 + cw]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(label.astype(np.float32)))[None],
                torch.from_numpy(np.ascontiguousarray(candidates.astype(np.float32)))[None])


# ---------------------------------------------------------------- metric


@torch.no_grad()
def precision_at_retention(model, loader, device, retention):
    """Precision when the top `retention` share of contour pixels per image is kept.

    This is the quantity the plan's conversion formula takes, so it is the only one worth
    watching: AUC over a 6%-positive problem said 0.876 for a selector that lost BF1.
    """
    model.eval()
    tp = sel = pos = cont = 0
    for x, y, c in loader:
        x, y, c = x.to(device), y.to(device), c.to(device)
        logit = model(x)
        for b in range(x.shape[0]):
            cm = c[b, 0] > 0.5
            n = int(cm.sum())
            if n == 0:
                continue
            keep = top_share_mask_torch(logit[b, 0], cm, retention)
            lab = (y[b, 0] > 0.5) & cm
            tp += int((keep & lab).sum()); sel += int(keep.sum())
            pos += int(lab.sum()); cont += n
    return {
        "precision": tp / max(sel, 1),
        "recall": tp / max(pos, 1),
        "base_rate": pos / max(cont, 1),
        "retention": sel / max(cont, 1),
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sources = mix_sources(args)
    tr_sets, va_sets = [], []
    for src in sources:
        sub = argparse.Namespace(**{**vars(args), **src})
        pairs = list_frames(sub)
        if src["max_images"]:
            # Deterministic subsample over the whole list, not the first N: the frames
            # are ordered by scene, so a head slice would keep only a few scenes.
            idx = np.random.default_rng(args.seed).permutation(len(pairs))[: src["max_images"]]
            pairs = [pairs[i] for i in sorted(idx)]

        # scene-level split: frames from one room are near-duplicates of each other
        scenes = sorted({p.parent.name for p, _ in pairs})
        n_val = (round(len(scenes) * args.val_frac) if args.val_frac else args.val_scenes)
        n_val = min(max(1, n_val), len(scenes) - 1)
        val_scenes = set(rng.choice(scenes, size=n_val, replace=False))
        tr = [p for p in pairs if p[0].parent.name not in val_scenes]
        va = [p for p in pairs if p[0].parent.name in val_scenes]
        print(f"[{src['dataset']}] frames {len(pairs)}  scenes {len(scenes)}  ->  "
              f"train {len(tr)} / val {len(va)} ({len(val_scenes)} scenes held out)")
        assert va and tr
        mk_set = lambda ps: ContourDataset(
            ps, Path(src["rgb_dir"]), src["pred_cache_dir"], src["mask_cache_dir"],
            args.t, args.label_px, no_sam=args.no_sam, no_rgb=args.no_rgb,
            dataset=src["dataset"], crop=args.crop, pred_space=src["pred_space"])
        tr_sets.append(mk_set(tr))
        va_sets.append(mk_set(va))

    cat = lambda ss: ss[0] if len(ss) == 1 else torch.utils.data.ConcatDataset(ss)
    mk = lambda ds, sh: torch.utils.data.DataLoader(
        cat(ds), batch_size=args.batch_size, shuffle=sh, num_workers=args.num_workers)
    dl_tr, dl_va = mk(tr_sets, True), mk(va_sets, False)
    if len(sources) > 1:
        print(f"mixed: train {sum(len(s) for s in tr_sets)} / "
              f"val {sum(len(s) for s in va_sets)} over {len(sources)} sources"
              f"{'' if args.crop is None else f'  crop {args.crop[0]}x{args.crop[1]}'}")

    in_ch = in_channels(no_rgb=args.no_rgb, no_sam=args.no_sam)
    model = UNet(in_ch, args.base_ch).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"UNet in_ch={in_ch} base={args.base_ch}  params={n_par/1e6:.2f}M"
          f"{'   [no_sam]' if args.no_sam else ''}{'   [no_rgb]' if args.no_rgb else ''}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    hist, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = nb = 0
        for x, y, c in tqdm(dl_tr, desc=f"ep{ep}", leave=False):
            x, y, c = x.to(device), y.to(device), c.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logit = model(x)
                # loss only where a contour exists: the other 88% of pixels are not a question
                per_px = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
                loss = (per_px * c).sum() / c.sum().clamp(min=1.0)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss); nb += 1
        sched.step()
        m = precision_at_retention(model, dl_va, device, args.retention)
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), **m})
        star = ""
        if m["precision"] > best:
            best = m["precision"]
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "val_precision": best, "epoch": ep}, out_dir / "best.pt")
            star = "  <- best"
        print(f"ep{ep:>3}  loss {tot/max(nb,1):.4f}  val 適合率 {m['precision']*100:5.1f}%  "
              f"再現率 {m['recall']*100:5.1f}%  (ベース {m['base_rate']*100:.2f}%, "
              f"残存 {m['retention']*100:.2f}%){star}")

    gain = 0.69 * (best * 100 - 37.1)
    summary = {"history": hist, "best_val_precision": best,
               "predicted_raw_bf1_gain_pct": gain,
               "reference_zero_training_precision": 0.346,
               "breakeven_precision": 0.371, "target_precision": 0.516}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n最良の val 適合率 {best*100:.1f}%")
    print(f"  学習ゼロの閾値 34.6% を超えたか : {'はい' if best > 0.346 else 'いいえ'}")
    print(f"  損益分岐 37.1% を超えたか       : {'はい' if best > 0.371 else 'いいえ'}")
    print(f"  目標 51.6% を超えたか           : {'はい' if best > 0.516 else 'いいえ'}")
    print(f"  換算した生の BF1 増分           : {gain:+.1f}%")
    print(f"\nSaved: {out_dir/'best.pt'}, {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
