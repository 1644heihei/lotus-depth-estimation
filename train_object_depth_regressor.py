#!/usr/bin/env python
"""Train object depth regressor on Hypersim train ONLY."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.object_depth_regressor import (
    BaselinePredictor,
    LinearDepthPredictor,
    ObjectDepthMLP,
    ObjectDepthTabularDataset,
    compute_depth_metrics,
    feature_dim_for_version,
    filter_training_items,
    image_level_split,
    load_records_from_manifest,
    predict_mlp_depth_m,
    records_to_arrays,
    scene_level_split,
    save_model_bundle,
)
from utils.object_depth_record import OBJECT_RECORD_SCHEMA_VERSION
from utils.seed_all import seed_all

logger = logging.getLogger(__name__)

FORBIDDEN_PATH_FRAGMENTS = ("nyuv2", "nyu_labeled", "nyu_depth")


def assert_hypersim_only(path: str, label: str) -> None:
    lower = path.replace("\\", "/").lower()
    for frag in FORBIDDEN_PATH_FRAGMENTS:
        if frag in lower:
            raise ValueError(f"{label} must not point to NYUv2/test data: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train object depth regressor (Hypersim train only).")
    p.add_argument("--detail_root", type=str, default="D:/lotus/data/object_depth_records_v2/hypersim_train")
    p.add_argument(
        "--manifest",
        type=str,
        default="D:/lotus/data/hypersim_yolo_detections/train_manifest_score0.5.json",
    )
    p.add_argument("--output_dir", type=str, default="output/object_depth_regressor_v3")
    p.add_argument("--roi_feature_cache", type=str, default=None, help="Optional .npz ROI visual features cache.")
    p.add_argument("--roi_projection_dim", type=int, default=64, help="Dimension to project ROI features down to.")
    p.add_argument("--feature_version", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--split_level", choices=["image", "scene"], default="image")
    p.add_argument("--min_depth_m", type=float, default=0.05)
    p.add_argument("--max_depth_m", type=float, default=30.0)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--huber_loss", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def evaluate_mlp(model, items, device, feature_version: int) -> dict:
    if not items:
        return compute_depth_metrics(np.array([]), np.array([]))
    class_ids, features, depths, roi_features = records_to_arrays(items, feature_version=feature_version)
    preds = []
    for i, (cid, feat) in enumerate(zip(class_ids, features)):
        roi_f = roi_features[i] if roi_features is not None else None
        preds.append(predict_mlp_depth_m(model, int(cid), feat, device, roi_features=roi_f))
    return compute_depth_metrics(np.array(preds), depths)


def evaluate_baseline(baseline: BaselinePredictor, items, per_class: bool) -> dict:
    preds = []
    gts = []
    for it in items:
        gts.append(it.record.depth_gt_m)
        preds.append(baseline.predict_depth(it.record.class_id) if per_class else baseline.global_median)
    return compute_depth_metrics(np.array(preds), np.array(gts))


def evaluate_linear(linear: LinearDepthPredictor, items, feature_version: int) -> dict:
    class_ids, features, depths, _ = records_to_arrays(items, feature_version=feature_version)
    preds = [float(np.exp(linear.predict_log_depth(int(cid), feat))) for cid, feat in zip(class_ids, features)]
    return compute_depth_metrics(np.array(preds), np.array(depths))


def evaluate_ensemble(model, linear, items, device, feature_version: int) -> dict:
    class_ids, features, depths, roi_features = records_to_arrays(items, feature_version=feature_version)
    preds = []
    for i, (cid, feat) in enumerate(zip(class_ids, features)):
        roi_f = roi_features[i] if roi_features is not None else None
        mlp_d = predict_mlp_depth_m(model, int(cid), feat, device, roi_features=roi_f)
        lin_d = float(np.exp(linear.predict_log_depth(int(cid), feat)))
        preds.append(float(np.exp(0.5 * (np.log(max(mlp_d, 1e-4)) + np.log(max(lin_d, 1e-4))))))
    return compute_depth_metrics(np.array(preds), depths)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    seed_all(args.seed)

    assert_hypersim_only(args.detail_root, "detail_root")
    assert_hypersim_only(args.manifest, "manifest")

    roi_cache = None
    if args.roi_feature_cache:
        from utils.roi_feature_extractor import RoiFeatureCache

        logger.info("Loading ROI feature cache from %s ...", args.roi_feature_cache)
        roi_cache = RoiFeatureCache.load(args.roi_feature_cache)

    items = load_records_from_manifest(args.manifest, args.detail_root, roi_cache=roi_cache)
    if not items:
        raise FileNotFoundError("No object records found.")
    before = len(items)
    items = filter_training_items(items, min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m)
    logger.info("Depth filter [%.2f, %.2f] m: %d -> %d records", args.min_depth_m, args.max_depth_m, before, len(items))

    split_fn = scene_level_split if args.split_level == "scene" else image_level_split
    train_items, val_items = split_fn(items, val_ratio=args.val_ratio, seed=args.seed)
    feat_dim = feature_dim_for_version(args.feature_version)
    has_roi = any(it.roi_feature is not None for it in train_items)
    roi_feat_dim = 512 if has_roi else 0

    logger.info(
        "feature_v=%d dim=%d roi_dim=%d records total=%d train=%d val=%d",
        args.feature_version,
        feat_dim,
        roi_feat_dim,
        len(items),
        len(train_items),
        len(val_items),
    )

    baseline = BaselinePredictor.fit(train_items)
    linear = LinearDepthPredictor.fit(train_items, feature_version=args.feature_version)

    train_ds = ObjectDepthTabularDataset(train_items, feature_version=args.feature_version)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ObjectDepthMLP(
        embed_dim=args.embed_dim,
        feature_dim=feat_dim,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
        roi_feature_dim=roi_feat_dim,
        roi_projection_dim=args.roi_projection_dim,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss() if args.huber_loss else nn.L1Loss()

    best_val = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            cid = batch["class_id"].to(device)
            feat = batch["features"].to(device)
            target = batch["log_depth"].to(device)
            roi_feat = batch.get("roi_features")
            if roi_feat is not None:
                roi_feat = roi_feat.to(device)
            pred = model(cid, feat, roi_features=roi_feat)
            loss = criterion(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))

        val_mlp = evaluate_mlp(model, val_items, device, args.feature_version)
        logger.info(
            "epoch=%d loss=%.4f val_abs_rel=%.4f val_delta1=%.4f",
            epoch,
            float(np.mean(losses)),
            val_mlp["abs_rel"],
            val_mlp["delta1"],
        )
        if val_mlp["abs_rel"] < best_val:
            best_val = val_mlp["abs_rel"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                logger.info("Early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_mlp = evaluate_mlp(model, val_items, device, args.feature_version)
    val_lin = evaluate_linear(linear, val_items, args.feature_version)
    val_ens = evaluate_ensemble(model, linear, val_items, device, args.feature_version)

    model_type = "mlp"
    if val_ens["abs_rel"] <= min(val_mlp["abs_rel"], val_lin["abs_rel"]):
        model_type = "ensemble"
    elif val_lin["abs_rel"] < val_mlp["abs_rel"]:
        model_type = "linear"

    metrics = {
        "train_data_source": "hypersim_train",
        "feature_version": args.feature_version,
        "depth_filter": [args.min_depth_m, args.max_depth_m],
        "split_level": args.split_level,
        "num_records_train": len(train_items),
        "num_records_val": len(val_items),
        "roi_feature_dim": roi_feat_dim,
        "roi_projection_dim": args.roi_projection_dim,
        "selected_model_type": model_type,
        "mlp_val": val_mlp,
        "linear_val": val_lin,
        "ensemble_val": val_ens,
        "baseline_per_class_val": evaluate_baseline(baseline, val_items, per_class=True),
    }

    config = {
        "train_data_source": "hypersim_train",
        "feature_version": args.feature_version,
        "feature_dim": feat_dim,
        "num_classes": 91,
        "embed_dim": args.embed_dim,
        "hidden": list(args.hidden),
        "dropout": args.dropout,
        "model_type": model_type,
        "detection_score_thr": 0.5,
        "depth_filter": [args.min_depth_m, args.max_depth_m],
        "split_seed": args.seed,
        "val_ratio": args.val_ratio,
        "split_level": args.split_level,
        # Resolution the RGB is resized to before ROI crops are cut for this regressor's
        # features. This is NOT the Lotus diffusion pipeline's inference resolution - they
        # are unrelated settings. eval_regressor_predepth_nyuv2.py used to fall back to this
        # value for the pipeline, which silently ran every experiment at 512 instead of
        # Lotus's official default of 768. See docs/phase0_findings.md 3.3.1.
        "roi_processing_res": 512,
        "processing_res": 512,  # deprecated alias, kept so existing configs still load
        "precision": "float32",
        "record_schema_version": OBJECT_RECORD_SCHEMA_VERSION,
        "roi_feature_dim": roi_feat_dim,
        "roi_projection_dim": args.roi_projection_dim,
    }
    save_model_bundle(args.output_dir, model, baseline, linear, config)
    split_manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "split_level": args.split_level,
        "train_images": sorted({item.rgb_path for item in train_items}),
        "val_images": sorted({item.rgb_path for item in val_items}),
    }
    (Path(args.output_dir) / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    (Path(args.output_dir) / "metrics_hypersim_holdout.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info("Saved %s model_type=%s metrics=%s", args.output_dir, model_type, json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
