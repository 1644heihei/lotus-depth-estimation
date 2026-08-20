"""Tabular object-depth regressor (class + bbox size -> depth)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.object_depth_record import ObjectDepthRecord, iter_object_record_files, load_object_records


NUM_CLASSES = 91  # COCO 0-90
FEATURE_DIM_V1 = 4
FEATURE_DIM_V2 = 7
FEATURE_DIM_V3 = 13


def extract_record_features(
    record: ObjectDepthRecord,
    *,
    feature_version: int = 1,
    score: Optional[float] = None,
) -> np.ndarray:
    w = float(record.bbox_w)
    h = float(record.bbox_h)
    area = float(record.bbox_area)
    cy = float(record.bbox_cy_norm)
    if feature_version <= 1:
        return np.array([w, h, area, cy], dtype=np.float32)
    aspect = w / max(h, 1e-6)
    log_area = float(np.log(max(area, 1e-6)))
    det_score = float(record.score if score is None else score)
    base = [w, h, area, cy, aspect, log_area, det_score]
    if feature_version == 2:
        return np.array(base, dtype=np.float32)
    mask_features = [
        float(record.mask_fill_ratio),
        float(np.log(max(record.mask_axis_ratio, 1.0))),
        float(record.mask_angle_sin2),
        float(record.mask_angle_cos2),
        float(record.mask_orientation_confidence),
        float(record.mask_valid),
    ]
    return np.array(base + mask_features, dtype=np.float32)


def feature_dim_for_version(feature_version: int) -> int:
    if feature_version <= 1:
        return FEATURE_DIM_V1
    if feature_version == 2:
        return FEATURE_DIM_V2
    return FEATURE_DIM_V3


def filter_training_items(
    items: Sequence[RecordWithImage],
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 30.0,
) -> List[RecordWithImage]:
    out: List[RecordWithImage] = []
    for it in items:
        d = it.record.depth_gt_m
        if d < min_depth_m or d > max_depth_m:
            continue
        out.append(it)
    return out


@dataclass
class RecordWithImage:
    record: ObjectDepthRecord
    rgb_path: str
    roi_feature: Optional[np.ndarray] = None


def records_from_detail_root(
    detail_root: str | Path,
    *,
    rgb_paths_filter: Optional[Sequence[str]] = None,
) -> List[RecordWithImage]:
    detail_root = Path(detail_root)
    allow = set(rgb_paths_filter) if rgb_paths_filter is not None else None
    out: List[RecordWithImage] = []
    for csv_path in iter_object_record_files(detail_root):
        stem = csv_path.name.replace("_objects.csv", "")
        parent = csv_path.parent.relative_to(detail_root)
        # Reconstruct a pseudo rgb path key for grouping; actual path from manifest when available.
        rgb_key = str(parent / f"{stem}.png")
        records = load_object_records(csv_path)
        for rec in records:
            out.append(RecordWithImage(record=rec, rgb_path=rgb_key))
    if allow is None:
        return out
    # When manifest gives full rgb paths, match by stem suffix.
    allow_stems = {Path(p).stem for p in allow}
    filtered: List[RecordWithImage] = []
    for item in out:
        if item.rgb_path in allow:
            filtered.append(item)
        elif Path(item.rgb_path).stem in allow_stems:
            filtered.append(item)
    return filtered


def load_records_from_manifest(
    manifest_path: str | Path,
    detail_root: str | Path,
    roi_cache: Optional[Any] = None,
) -> List[RecordWithImage]:
    from utils.detail_train_dataset import load_detail_train_manifest

    rgb_paths, _, _ = load_detail_train_manifest(manifest_path)
    detail_root = Path(detail_root)
    out: List[RecordWithImage] = []
    from utils.object_depth_record import load_object_records_for_rgb, objects_csv_path

    for rgb_path in rgb_paths:
        csv_path = objects_csv_path(rgb_path, detail_root)
        if not csv_path.is_file():
            continue
        records = load_object_records(csv_path)
        roi_feats = None
        if roi_cache is not None:
            roi_feats = roi_cache.get_features(rgb_path)
        for i, rec in enumerate(records):
            roi_f = roi_feats[i] if (roi_feats is not None and i < len(roi_feats)) else None
            out.append(RecordWithImage(record=rec, rgb_path=str(rgb_path), roi_feature=roi_f))
    return out


def image_level_split(
    items: Sequence[RecordWithImage],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[RecordWithImage], List[RecordWithImage]]:
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[RecordWithImage]] = {}
    for item in items:
        groups.setdefault(item.rgb_path, []).append(item)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val])
    train_items: List[RecordWithImage] = []
    val_items: List[RecordWithImage] = []
    for key, group in groups.items():
        if key in val_keys:
            val_items.extend(group)
        else:
            train_items.extend(group)
    return train_items, val_items


def scene_level_split(
    items: Sequence[RecordWithImage],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[RecordWithImage], List[RecordWithImage]]:
    """Keep every frame from a Hypersim scene in the same split."""
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[RecordWithImage]] = {}
    for item in items:
        parts = Path(item.rgb_path.replace("\\", "/")).parts
        scene_id = next(
            (part for part in reversed(parts[:-1]) if part.startswith("ai_")),
            Path(item.rgb_path).parent.name,
        )
        groups.setdefault(scene_id, []).append(item)
    keys = sorted(groups)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:n_val])
    train_items = [
        item for key, group in groups.items() if key not in val_keys for item in group
    ]
    val_items = [
        item for key, group in groups.items() if key in val_keys for item in group
    ]
    return train_items, val_items


def records_to_arrays(
    items: Sequence[RecordWithImage],
    *,
    feature_version: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    class_ids = np.array([it.record.class_id for it in items], dtype=np.int64)
    features = np.stack([extract_record_features(it.record, feature_version=feature_version) for it in items])
    depths = np.array([it.record.depth_gt_m for it in items], dtype=np.float32)
    has_roi = any(it.roi_feature is not None for it in items)
    roi_features = None
    if has_roi:
        roi_features_list = []
        feat_dim = next((it.roi_feature.shape[0] for it in items if it.roi_feature is not None), 512)
        for it in items:
            if it.roi_feature is not None:
                roi_features_list.append(it.roi_feature)
            else:
                roi_features_list.append(np.zeros(feat_dim, dtype=np.float32))
        roi_features = np.stack(roi_features_list).astype(np.float32)
    return class_ids, features, depths, roi_features


class ObjectDepthTabularDataset(Dataset):
    def __init__(self, items: Sequence[RecordWithImage], *, feature_version: int = 1):
        self.feature_version = feature_version
        self.class_ids, self.features, self.depths, self.roi_features = records_to_arrays(
            items, feature_version=feature_version
        )

    def __len__(self) -> int:
        return int(self.depths.shape[0])

    def __getitem__(self, idx: int):
        data = {
            "class_id": torch.tensor(self.class_ids[idx], dtype=torch.long),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "depth_m": torch.tensor(self.depths[idx], dtype=torch.float32),
            "log_depth": torch.tensor(np.log(max(float(self.depths[idx]), 1e-4)), dtype=torch.float32),
        }
        if self.roi_features is not None:
            data["roi_features"] = torch.tensor(self.roi_features[idx], dtype=torch.float32)
        return data


class ObjectDepthMLP(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = 16,
        feature_dim: int = 4,
        hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        roi_feature_dim: int = 0,
        roi_projection_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.hidden = hidden
        self.dropout = dropout
        self.roi_feature_dim = roi_feature_dim
        self.roi_projection_dim = roi_projection_dim

        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.roi_projection = None
        roi_out_dim = 0
        if roi_feature_dim > 0:
            self.roi_projection = nn.Sequential(
                nn.Linear(roi_feature_dim, roi_projection_dim),
                nn.ReLU(),
                nn.LayerNorm(roi_projection_dim),
            )
            roi_out_dim = roi_projection_dim

        layers: List[nn.Module] = []
        in_dim = embed_dim + feature_dim + roi_out_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        class_id: torch.Tensor,
        features: torch.Tensor,
        roi_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.class_embed(class_id.clamp(0, self.class_embed.num_embeddings - 1))
        parts = [emb, features]
        if self.roi_projection is not None:
            if roi_features is None:
                # Zero padding fallback if roi_features not passed
                batch_size = class_id.shape[0]
                zeros_roi = class_id.new_zeros(
                    (batch_size, self.roi_feature_dim), dtype=features.dtype
                )
                parts.append(self.roi_projection(zeros_roi))
            else:
                parts.append(self.roi_projection(roi_features))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)


@dataclass
class BaselinePredictor:
    global_median: float
    per_class_median: Dict[int, float]

    @classmethod
    def fit(cls, items: Sequence[RecordWithImage]) -> "BaselinePredictor":
        depths = [it.record.depth_gt_m for it in items]
        global_median = float(np.median(depths))
        per_class: Dict[int, List[float]] = {}
        for it in items:
            per_class.setdefault(it.record.class_id, []).append(it.record.depth_gt_m)
        per_class_median = {k: float(np.median(v)) for k, v in per_class.items()}
        return cls(global_median=global_median, per_class_median=per_class_median)

    def predict_depth(self, class_id: int) -> float:
        return float(self.per_class_median.get(class_id, self.global_median))


@dataclass
class LinearDepthPredictor:
    weights: np.ndarray  # [num_classes + feat_dim + 1] last is bias
    num_classes: int = NUM_CLASSES
    feature_dim: int = FEATURE_DIM_V1

    @classmethod
    def fit(
        cls,
        items: Sequence[RecordWithImage],
        *,
        ridge: float = 1e-3,
        feature_version: int = 1,
    ) -> "LinearDepthPredictor":
        feat_dim = feature_dim_for_version(feature_version)
        class_ids, features, depths, _ = records_to_arrays(items, feature_version=feature_version)
        n = len(depths)
        x = np.zeros((n, NUM_CLASSES + feat_dim), dtype=np.float64)
        for i, cid in enumerate(class_ids):
            x[i, int(cid)] = 1.0
            x[i, NUM_CLASSES:] = features[i]
        y = np.log(np.clip(depths, 1e-4, None))
        xb = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
        if ridge > 0:
            reg = ridge * np.eye(xb.shape[1], dtype=np.float64)
            reg[-1, -1] = 0.0
            w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
        else:
            w = np.linalg.lstsq(xb, y, rcond=None)[0]
        return cls(weights=w.astype(np.float32), feature_dim=feat_dim)

    def predict_log_depth(self, class_id: int, features: np.ndarray) -> float:
        feat_dim = self.feature_dim
        x = np.zeros(NUM_CLASSES + feat_dim, dtype=np.float32)
        x[int(class_id)] = 1.0
        x[NUM_CLASSES : NUM_CLASSES + feat_dim] = features.astype(np.float32)[:feat_dim]
        xb = np.concatenate([x, np.array([1.0], dtype=np.float32)])
        return float(xb @ self.weights)


def compute_depth_metrics(pred_m: np.ndarray, gt_m: np.ndarray) -> Dict[str, float]:
    pred_m = np.asarray(pred_m, dtype=np.float64)
    gt_m = np.asarray(gt_m, dtype=np.float64)
    valid = np.isfinite(pred_m) & np.isfinite(gt_m) & (gt_m > 1e-4) & (pred_m > 1e-4)
    if valid.sum() == 0:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "count": 0}
    p = pred_m[valid]
    g = gt_m[valid]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    return {"abs_rel": abs_rel, "rmse": rmse, "delta1": delta1, "count": int(valid.sum())}


@torch.no_grad()
def predict_mlp_depth_m(
    model: ObjectDepthMLP,
    class_id: int,
    features: np.ndarray,
    device: torch.device,
    roi_features: Optional[np.ndarray] = None,
) -> float:
    model.eval()
    cid = torch.tensor([class_id], dtype=torch.long, device=device)
    feat = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    roi_t = None
    if roi_features is not None:
        roi_t = torch.tensor(roi_features, dtype=torch.float32, device=device).unsqueeze(0)
    log_d = model(cid, feat, roi_features=roi_t).item()
    return float(np.exp(log_d))


class ObjectDepthRegressorBundle:
    """Loadable wrapper for inference (Step 2)."""

    def __init__(
        self,
        mlp: ObjectDepthMLP,
        baseline: BaselinePredictor,
        linear: Optional[LinearDepthPredictor],
        config: dict,
        device: torch.device,
    ):
        self.mlp = mlp
        self.baseline = baseline
        self.linear = linear
        self.config = config
        self.device = device
        self.model_type = config.get("model_type", "mlp")
        self._roi_extractor = None

    @property
    def roi_extractor(self):
        if self._roi_extractor is None and self.config.get("roi_feature_dim", 0) > 0:
            from utils.roi_feature_extractor import RoiFeatureExtractor

            self._roi_extractor = RoiFeatureExtractor(device=self.device)
        return self._roi_extractor

    @classmethod
    def load(cls, model_dir: str | Path, device: Optional[torch.device] = None) -> "ObjectDepthRegressorBundle":
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp = ObjectDepthMLP(
            num_classes=config.get("num_classes", NUM_CLASSES),
            embed_dim=config.get("embed_dim", 16),
            feature_dim=config.get("feature_dim", FEATURE_DIM_V1),
            hidden=tuple(config.get("hidden", [128, 64])),
            dropout=config.get("dropout", 0.1),
            roi_feature_dim=config.get("roi_feature_dim", 0),
            roi_projection_dim=config.get("roi_projection_dim", 64),
        )
        state = torch.load(model_dir / "model.pt", map_location=device)
        mlp.load_state_dict(state)
        mlp.to(device)
        mlp.eval()
        baseline = BaselinePredictor(
            global_median=config["baseline"]["global_median"],
            per_class_median={int(k): float(v) for k, v in config["baseline"]["per_class_median"].items()},
        )
        linear = None
        if (model_dir / "linear_weights.npy").is_file():
            w = np.load(model_dir / "linear_weights.npy")
            linear = LinearDepthPredictor(
                weights=w,
                feature_dim=config.get("feature_dim", feature_dim_for_version(config.get("feature_version", 1))),
            )
        return cls(mlp=mlp, baseline=baseline, linear=linear, config=config, device=device)

    def predict_depth_m(
        self,
        class_id: int,
        bbox_w: float,
        bbox_h: float,
        bbox_cy_norm: Optional[float] = None,
        score: float = 0.5,
        *,
        mask_fill_ratio: float = 1.0,
        mask_axis_ratio: float = 1.0,
        mask_angle_sin2: float = 0.0,
        mask_angle_cos2: float = 1.0,
        mask_orientation_confidence: float = 0.0,
        mask_valid: float = 0.0,
        roi_features: Optional[np.ndarray] = None,
        rgb_np: Optional[np.ndarray] = None,
        bbox_pixels: Optional[Sequence[Union[int, float]]] = None,
    ) -> float:
        cy = 0.5 if bbox_cy_norm is None else bbox_cy_norm
        record = ObjectDepthRecord(
            class_id=class_id,
            class_name="",
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            bbox_area=bbox_w * bbox_h,
            bbox_cy_norm=cy,
            depth_gt_m=1.0,
            disparity_gt=1.0,
            score=score,
            x1=int(bbox_pixels[0]) if bbox_pixels is not None else 0,
            y1=int(bbox_pixels[1]) if bbox_pixels is not None else 0,
            x2=int(bbox_pixels[2]) if bbox_pixels is not None else 1,
            y2=int(bbox_pixels[3]) if bbox_pixels is not None else 1,
            mask_fill_ratio=mask_fill_ratio,
            mask_axis_ratio=mask_axis_ratio,
            mask_angle_sin2=mask_angle_sin2,
            mask_angle_cos2=mask_angle_cos2,
            mask_orientation_confidence=mask_orientation_confidence,
            mask_valid=mask_valid,
        )
        return self.predict_record(
            record,
            score=score,
            roi_features=roi_features,
            rgb_np=rgb_np,
            bbox_pixels=bbox_pixels,
        )

    def predict_record(
        self,
        record: ObjectDepthRecord,
        *,
        score: Optional[float] = None,
        roi_features: Optional[np.ndarray] = None,
        rgb_np: Optional[np.ndarray] = None,
        bbox_pixels: Optional[Sequence[Union[int, float]]] = None,
    ) -> float:
        """Predict from one canonical record so train/eval/inference share features."""
        fv = int(self.config.get("feature_version", 1))
        features = extract_record_features(record, feature_version=fv, score=score)
        class_id = int(record.class_id)
        if self.model_type == "baseline_global":
            return self.baseline.global_median
        if self.model_type == "baseline_per_class":
            return self.baseline.predict_depth(class_id)

        # Extract ROI feature if needed and available
        if roi_features is None and self.config.get("roi_feature_dim", 0) > 0:
            if rgb_np is not None:
                pixels = bbox_pixels if bbox_pixels is not None else [record.x1, record.y1, record.x2, record.y2]
                if self.roi_extractor is not None:
                    roi_features = self.roi_extractor.extract_single(rgb_np, pixels)

        mlp_depth = predict_mlp_depth_m(self.mlp, class_id, features, self.device, roi_features=roi_features)
        if self.model_type == "linear" and self.linear is not None:
            return float(np.exp(self.linear.predict_log_depth(class_id, features)))
        if self.model_type == "ensemble" and self.linear is not None:
            lin_depth = float(np.exp(self.linear.predict_log_depth(class_id, features)))
            return float(np.exp(0.5 * (np.log(max(mlp_depth, 1e-4)) + np.log(max(lin_depth, 1e-4)))))
        return mlp_depth


def save_model_bundle(
    output_dir: str | Path,
    mlp: ObjectDepthMLP,
    baseline: BaselinePredictor,
    linear: LinearDepthPredictor,
    config: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mlp.state_dict(), output_dir / "model.pt")
    np.save(output_dir / "linear_weights.npy", linear.weights)
    config = dict(config)
    config["roi_feature_dim"] = getattr(mlp, "roi_feature_dim", 0)
    config["roi_projection_dim"] = getattr(mlp, "roi_projection_dim", 64)
    config["baseline"] = {
        "global_median": baseline.global_median,
        "per_class_median": {str(k): v for k, v in baseline.per_class_median.items()},
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
