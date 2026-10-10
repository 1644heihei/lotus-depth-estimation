"""Where each dataset's frames, contours and cached predictions live.

Four scripts used to carry their own hard-coded defaults, and they disagreed: the
evaluator defaulted to Lotus predictions at 768, the apply script to Marigold V2 at 640,
the trainer to a third directory. Every experiment outside NYUv2 therefore retyped four
or five long paths from memory, and that is exactly how a Hypersim sweep in this project
was first launched against the wrong RGB directory at the wrong resolution.

So the paths live here once, keyed by what actually determines them:

  frames and contours   by (dataset, split) - the RGB tree and its SAM masks
  predictions           by (dataset, split, model) - a prediction belongs to a model,
                        and its processing resolution is part of where it was written

`resolve()` fills in whatever the caller left unset, so passing a flag still overrides
the registry and nothing that worked before changes. `verify()` checks the registry
against the disk, because a registry that is wrong is worse than retyping: it is trusted.
"""

from __future__ import annotations

from pathlib import Path

_NYU = ("C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/"
        "nyu_labeled_extracted.tar")
_REPO = Path(__file__).resolve().parents[1]

# The split each dataset is normally used at, so --dataset alone is enough.
DEFAULT_SPLIT = {"nyuv2": "test", "scannet": "val", "hypersim": "eval",
                 "hypersim10k": "train"}

RGB_DIRS = {
    ("nyuv2", "test"): f"{_NYU}/test",
    ("nyuv2", "train"): f"{_NYU}/train",
    ("scannet", "val"): "D:/lotus/data/scannet_rgb_only/val",
    ("hypersim", "eval"): "D:/lotus/data/hypersim_holdout_rgb/eval",
    ("hypersim10k", "train"): "D:/lotus/data/hypersim_sel10k_rgb/train",
}

SAM_DIRS = {
    ("nyuv2", "test"): "D:/lotus/data/oracle_cache/sam_auto48",
    ("nyuv2", "train"): "D:/lotus/data/oracle_cache/sam_auto48_train",
    ("scannet", "val"): "D:/lotus/data/oracle_cache/sam_auto48_scannet",
    ("hypersim", "eval"): "D:/lotus/data/oracle_cache/sam_auto48_hypersim",
    ("hypersim10k", "train"): "D:/lotus/data/oracle_cache/sam_auto48_hs10k",
}

# (dir, processing_res). The evaluator appends res<N>; the apply and training scripts
# want the full path, which `pred_dir()` returns.
PRED_CACHES = {
    ("nyuv2", "test", "marigold_v2"): ("D:/lotus/data/marigold_v2_pred", 640),
    ("nyuv2", "train", "marigold_v2"): ("D:/lotus/data/marigold_v2_pred_train", 640),
    ("nyuv2", "test", "lotus_v1"): ("D:/lotus/data/oracle_cache/lotus_pred", 768),
    ("scannet", "val", "marigold_v2"): ("D:/lotus/data/marigold_v2_pred_scannet", 640),
    ("hypersim", "eval", "marigold_v2"): ("D:/lotus/data/marigold_v2_pred_hypersim", 1024),
    ("hypersim10k", "train", "marigold_v2"): ("D:/lotus/data/marigold_v2_pred_hs10k", 1024),
}

# What each model's cached prediction holds. Feeding disparity in raw inverts the signal,
# so this travels with the prediction rather than being passed separately.
PRED_SPACE = {"marigold_v2": "log", "lotus_v1": "disparity"}
ALIGN_SPACE = {"marigold_v2": "log", "lotus_v1": "disparity"}

EXTRA = {
    "scannet": {"scannet_root": "D:/lotus/data/marigold_depth_eval/scannet"},
    "hypersim": {"depth_root": "D:/lotus/data/hypersim_processed/train",
                 "holdout_split": str(_REPO / "datasets/hypersim_holdout.json")},
    "hypersim10k": {"hs10k_split": str(_REPO / "datasets/hypersim_selector_train_10k.json")},
}

MODELS = tuple(sorted({m for _, _, m in PRED_CACHES}))


def splits_of(dataset: str) -> tuple:
    return tuple(s for d, s in RGB_DIRS if d == dataset)


def pred_dir(dataset: str, split: str, model: str) -> str:
    """The directory an apply or training script reads, resolution included."""
    root, res = PRED_CACHES[(dataset, split, model)]
    return f"{root}/res{res}"


def resolve(args, parser=None, model: str | None = None, split: str | None = None):
    """Fill the registry's paths in, without overriding anything explicitly passed.

    "Explicitly passed" is decided against the parser's own defaults: a value still equal
    to its default was not typed by the caller and may be replaced. This matters because
    the historical defaults are NOT empty - they point at NYUv2 and at Marigold V2 - so a
    rule of "fill only what is unset" would leave `--dataset hypersim --model marigold_v2`
    reading NYUv2 frames and never say so. That is the failure this registry exists to
    stop, so it has to beat a default while still losing to a flag.

    Without `parser` it falls back to filling only unset values, which is right for
    callers that build a namespace themselves.
    """
    ds = args.dataset
    split = split or getattr(args, "split", None) or DEFAULT_SPLIT[ds]
    model = model or getattr(args, "model", None)

    def is_default(name):
        if parser is None:
            return not getattr(args, name, None)
        return getattr(args, name, None) == parser.get_default(name)

    def fill(name, value):
        if value is not None and hasattr(args, name) and is_default(name):
            setattr(args, name, value)

    fill("rgb_dir", RGB_DIRS.get((ds, split)))
    fill("mask_cache_dir", SAM_DIRS.get((ds, split)))
    for k, v in EXTRA.get(ds, {}).items():
        fill(k, v)

    if model:
        entry = PRED_CACHES.get((ds, split, model))
        assert entry, (f"no cached {model} predictions for {ds}/{split}; "
                       f"have {sorted(k for k in PRED_CACHES if k[0] == ds)}")
        root, res = entry
        # The evaluator keeps the resolution in its own flag and appends res<N> itself;
        # every other script wants the joined path.
        if hasattr(args, "processing_res"):
            fill("pred_cache_dir", root)
            fill("processing_res", res)
        else:
            fill("pred_cache_dir", f"{root}/res{res}")
        fill("pred_space", PRED_SPACE.get(model))
        fill("align_space", ALIGN_SPACE.get(model))
    return args


def verify() -> list:
    """Every registered path that is not on disk. Empty list means the registry is true."""
    bad = []
    for (d, s), v in RGB_DIRS.items():
        if not Path(v).is_dir():
            bad.append(f"RGB {d}/{s}: {v}")
    for (d, s), v in SAM_DIRS.items():
        if not Path(v).is_dir():
            bad.append(f"SAM {d}/{s}: {v}")
    for (d, s, m), (v, res) in PRED_CACHES.items():
        if not (Path(v) / f"res{res}").is_dir():
            bad.append(f"PRED {d}/{s}/{m}: {v}/res{res}")
    for d, kv in EXTRA.items():
        for k, v in kv.items():
            if not Path(v).exists():
                bad.append(f"{k} {d}: {v}")
    return bad


if __name__ == "__main__":
    missing = verify()
    print("\n".join(missing) if missing else "all registered paths exist")
    raise SystemExit(1 if missing else 0)
