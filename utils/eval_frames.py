"""Frame enumeration and validity masks, per evaluation dataset.

Every measurement in docs/ was taken on NYUv2 through one scoring path, and adding a second
dataset must not fork it. So the parts that actually differ between datasets live here and
the scripts keep their single code path:

  frames        NYUv2 pairs rgb_NNNN.png with depth_NNNN.png in the same folder; Hypersim
                reads the scene-level hold-out split and pairs each staged RGB with
                depth_plane_* under the untouched data root.
  depth range   NYUv2 is a 10 m indoor sensor; Hypersim is synthetic and Marigold V1's
                repackaging caps it at 65 m.
  crop          NYUv2 evaluation conventionally uses the Eigen crop. Hypersim has no such
                convention and needs none - its ground truth is exact everywhere.

Both store depth as uint16 millimetres, so the /1000 lives in the callers unchanged.

Hypersim needs two roots because inference runs over a staging tree holding only the RGB
files (scripts/infer.py globs every .png under image_dir, and depth_plane_* sits beside
rgb_* in the original layout). Cache paths are keyed off the staging root, which is also
what build_sam_masks.py was pointed at, so the two agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATASETS = ("nyuv2", "hypersim", "hypersim10k", "scannet")
MAX_DEPTH = {"nyuv2": 10.0, "hypersim": 65.0, "hypersim10k": 65.0, "scannet": 10.0}
MIN_DEPTH = {"nyuv2": 1e-3, "hypersim": 1e-3, "hypersim10k": 1e-3, "scannet": 1e-3}
# uint16 PNG -> metres. Marigold's repackaging uses 1000 for all three; KITTI, when added,
# needs 256, which is why this is a table rather than a constant in the callers.
DEPTH_SCALE = {"nyuv2": 1000.0, "hypersim": 1000.0, "hypersim10k": 1000.0,
               "scannet": 1000.0}
HS10K_SPLIT = "datasets/hypersim_selector_train_10k.json"
SCANNET_SPLIT = ("D:/lotus/marigold-v2/evaluation/data_split/scannet_depth/"
                 "scannet_val_sampled_list_800_1.txt")


def add_dataset_args(parser) -> None:
    """Add --dataset and the two paths only Hypersim needs."""
    parser.add_argument("--dataset", choices=DATASETS, default="nyuv2")
    parser.add_argument(
        "--holdout_split", type=str, default="datasets/hypersim_holdout.json",
        help="Hypersim only: the scene-level split whose eval_frames are scored.")
    parser.add_argument(
        "--depth_root", type=str, default="D:/lotus/data/hypersim_processed/train",
        help="Hypersim only: where depth_plane_* lives, since --rgb_dir is an RGB-only "
             "staging tree.")
    parser.add_argument(
        "--hs10k_split", type=str, default=HS10K_SPLIT,
        help="hypersim10k only: the 10,000-frame selector training list, which excludes "
             "the 20 scenes reserved in hypersim_holdout.json.")
    parser.add_argument(
        "--scannet_split", type=str, default=SCANNET_SPLIT,
        help="ScanNet only: the two-column list of rgb and depth paths to score.")
    parser.add_argument(
        "--scannet_root", type=str, default="D:/lotus/data/marigold_depth_eval/scannet",
        help="ScanNet only: where the split's paths resolve against, since --rgb_dir is "
             "an RGB-only staging tree.")


def list_frames(args) -> list[tuple[Path, Path]]:
    """[(rgb_path, depth_path)] for the configured dataset, in a stable order."""
    rgb_dir = Path(args.rgb_dir)
    if args.dataset == "nyuv2":
        from eval_regressor_predepth_nyuv2 import list_nyu_pairs
        return list_nyu_pairs(rgb_dir)

    if args.dataset == "hypersim10k":
        # Same layout as hypersim, a different frame list: depth_plane_* beside rgb_* in the
        # original root, rgb staged on its own so infer.py's glob sees only images.
        spec = json.loads(Path(args.hs10k_split).read_text(encoding="utf-8"))
        depth_root = Path(spec["data_root"])
        out = []
        for rel in spec["frames"]:
            rel = Path(rel)
            rgb = rgb_dir / rel
            depth = depth_root / rel.parent / rel.name.replace("rgb_", "depth_plane_")
            if rgb.is_file() and depth.is_file():
                out.append((rgb, depth))
        return out

    if args.dataset == "scannet":
        # The split names rgb and depth explicitly, so there is no name mangling to do;
        # rgb resolves against the staging tree and depth against the original download.
        root = Path(args.scannet_root)
        out = []
        for line in Path(args.scannet_split).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r, d = line.split()
            rgb, depth = rgb_dir / r, root / d
            if rgb.is_file() and depth.is_file():
                out.append((rgb, depth))
        return out

    depth_root = Path(args.depth_root)
    rels = json.loads(Path(args.holdout_split).read_text(encoding="utf-8"))["eval_frames"]
    out = []
    for rel in rels:
        rel = Path(rel)
        rgb = rgb_dir / rel
        depth = depth_root / rel.parent / rel.name.replace("rgb_", "depth_plane_")
        if rgb.is_file() and depth.is_file():
            out.append((rgb, depth))
    return out


def depth_scale(dataset: str) -> float:
    """Divisor that turns the stored uint16 depth into metres."""
    return DEPTH_SCALE[dataset]


def valid_mask(dataset: str, gt: np.ndarray) -> np.ndarray:
    """Pixels the metrics are averaged over."""
    m = np.isfinite(gt) & (gt > MIN_DEPTH[dataset]) & (gt < MAX_DEPTH[dataset])
    if dataset == "nyuv2":
        from eval_regressor_predepth_nyuv2 import eigen_valid_mask
        m &= eigen_valid_mask(*gt.shape)
    return m
