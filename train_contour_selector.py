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

Two constraints the plan derived the hard way:

  - The prediction is affine-invariant log depth, so its gradients carry an unknown
    per-image scale. Inputs are standardised per image, otherwise train and inference see
    different magnitudes.
  - Scattering the selection scores -20.35%, ten times worse than using every contour,
    because an isolated contour pixel is not a barrier at all. A convolutional model is
    smooth by construction, and a component-size floor is applied on top.

Trains on NYUv2 train (795 frames), validates on a scene-disjoint split of it, and never
touches the 654-frame test set - that is scored separately once, by the sharpening pipeline.
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
from scipy import ndimage as ndi
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_contour_feature_auc import load_contour
from eval_mask_contour_localization import discontinuities
from eval_object_oracle_ceiling import _cache_path
from eval_regressor_predepth_nyuv2 import eigen_valid_mask, list_nyu_pairs

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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--retention", type=float, default=0.06,
                   help="Fraction of contour pixels kept when scoring precision, matched "
                        "to what the end-to-end runs used.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- data


class ContourDataset(torch.utils.data.Dataset):
    """Five input channels and a label defined only on contour pixels.

    The prediction is standardised per image: Log-stage2 emits affine-invariant log depth,
    so its absolute level and scale mean nothing and would differ between training (where
    a prediction could be aligned to GT) and inference (where it cannot be).
    """

    def __init__(self, pairs, rgb_dir, pred_dir, mask_dir, t, label_px):
        self.pairs = list(pairs)
        self.rgb_dir, self.pred_dir, self.mask_dir = Path(rgb_dir), Path(pred_dir), Path(mask_dir)
        self.t, self.label_px = t, label_px

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rgb_path, depth_path = self.pairs[i]
        gt = np.array(Image.open(depth_path)).astype(np.float64) / 1000.0
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 10.0) & eigen_valid_mask(h, w)

        pred = np.load(_cache_path(rgb_path, self.rgb_dir, self.pred_dir, "_pred.npy")).astype(np.float32)
        m, s = float(pred.mean()), float(pred.std())
        pred = (pred - m) / (s + 1e-6)          # per-image: the affine is unknowable

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        contour = load_contour(_cache_path(rgb_path, self.rgb_dir, self.mask_dir, "_seg.npz"),
                               h, w) & valid

        dist = ndi.distance_transform_edt(~discontinuities(gt, valid, self.t))
        label = dist <= self.label_px

        x = np.concatenate([rgb.transpose(2, 0, 1),
                            pred[None],
                            contour.astype(np.float32)[None] * 2.0 - 1.0], axis=0)
        return (torch.from_numpy(x),
                torch.from_numpy(label.astype(np.float32))[None],
                torch.from_numpy(contour.astype(np.float32))[None])


# ---------------------------------------------------------------- model


def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four levels, one output logit per pixel. Small: the label is 6% positive on 6% of
    the image, so capacity is not the binding constraint - precision in the top few
    percent is."""

    def __init__(self, in_ch=5, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = block(in_ch, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.mid = block(c[2], c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.d3 = block(c[2] * 2, c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = block(c[1] * 2, c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = block(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


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
            k = max(int(round(n * retention)), 1)
            s = logit[b, 0][cm]
            thr = torch.topk(s, k).values[-1]
            keep = cm & (logit[b, 0] >= thr)
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

    rgb_dir = Path(args.rgb_dir)
    pairs = list_nyu_pairs(rgb_dir)
    if args.max_images:
        pairs = pairs[: args.max_images]

    # scene-level split: frames from one room are near-duplicates of each other
    scenes = sorted({p.parent.name for p, _ in pairs})
    val_scenes = set(rng.choice(scenes, size=min(args.val_scenes, len(scenes)), replace=False))
    tr = [p for p in pairs if p[0].parent.name not in val_scenes]
    va = [p for p in pairs if p[0].parent.name in val_scenes]
    print(f"frames {len(pairs)}  scenes {len(scenes)}  ->  train {len(tr)} / val {len(va)} "
          f"({len(val_scenes)} scenes held out)")
    assert va and tr

    mk = lambda ps, sh: torch.utils.data.DataLoader(
        ContourDataset(ps, rgb_dir, args.pred_cache_dir, args.mask_cache_dir,
                       args.t, args.label_px),
        batch_size=args.batch_size, shuffle=sh, num_workers=args.num_workers)
    dl_tr, dl_va = mk(tr, True), mk(va, False)

    model = UNet(5, args.base_ch).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"UNet base={args.base_ch}  params={n_par/1e6:.2f}M")
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
