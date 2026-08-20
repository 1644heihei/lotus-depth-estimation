"""Object-level cross-attention conditioning for Lotus-D."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

from utils.object_detection_cache import ObjectDetection


OBJECT_CONTINUOUS_FEATURE_DIM = 14


def normalize_image_key(path: str | Path) -> str:
    return str(path).replace("\\", "/").lower()


def pad_object_rows(
    class_rows: np.ndarray,
    feature_rows: np.ndarray,
    max_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_objects <= 0:
        raise ValueError("max_objects must be positive")
    class_rows = np.asarray(class_rows, dtype=np.int64)
    feature_rows = np.asarray(feature_rows, dtype=np.float32).reshape(
        -1, OBJECT_CONTINUOUS_FEATURE_DIM
    )
    if len(class_rows) != len(feature_rows):
        raise ValueError("class/features length mismatch")
    count = min(len(class_rows), max_objects)
    class_ids = np.zeros(max_objects, dtype=np.int64)
    features = np.zeros(
        (max_objects, OBJECT_CONTINUOUS_FEATURE_DIM), dtype=np.float32
    )
    mask = np.zeros(max_objects, dtype=bool)
    if count:
        class_ids[:count] = class_rows[:count]
        features[:count] = feature_rows[:count]
        mask[:count] = True
    return class_ids, features, mask


def detection_attention_features(
    detection: ObjectDetection,
    image_width: int,
    image_height: int,
    predicted_depth_m: float,
) -> np.ndarray:
    x1, y1, x2, y2 = detection.bbox
    width = max(float(x2 - x1) / max(image_width, 1), 1e-6)
    height = max(float(y2 - y1) / max(image_height, 1), 1e-6)
    cx = float(x1 + x2) * 0.5 / max(image_width, 1)
    cy = float(y1 + y2) * 0.5 / max(image_height, 1)
    return np.asarray(
        [
            cx,
            cy,
            width,
            height,
            np.log(max(width * height, 1e-6)),
            np.log(max(width / height, 1e-6)),
            float(detection.score),
            float(detection.mask_fill_ratio),
            np.log(max(float(detection.mask_axis_ratio), 1.0)),
            float(detection.mask_angle_sin2),
            float(detection.mask_angle_cos2),
            float(detection.mask_orientation_confidence),
            float(detection.mask_valid),
            np.log(max(float(predicted_depth_m), 1e-4)),
        ],
        dtype=np.float32,
    )


def object_rows_from_detections(
    detections: Sequence[ObjectDetection],
    image_width: int,
    image_height: int,
    regressor,
    rgb_np: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    class_ids: list[int] = []
    features: list[np.ndarray] = []
    for detection in sorted(
        detections, key=lambda item: item.score, reverse=True
    ):
        x1, y1, x2, y2 = detection.bbox
        bbox_w = float(x2 - x1) / max(image_width, 1)
        bbox_h = float(y2 - y1) / max(image_height, 1)
        bbox_cy = float(y1 + y2) * 0.5 / max(image_height, 1)
        predicted_depth = regressor.predict_depth_m(
            detection.class_id,
            bbox_w,
            bbox_h,
            bbox_cy,
            score=float(detection.score),
            mask_fill_ratio=float(detection.mask_fill_ratio),
            mask_axis_ratio=float(detection.mask_axis_ratio),
            mask_angle_sin2=float(detection.mask_angle_sin2),
            mask_angle_cos2=float(detection.mask_angle_cos2),
            mask_orientation_confidence=float(
                detection.mask_orientation_confidence
            ),
            mask_valid=float(detection.mask_valid),
            rgb_np=rgb_np,
            bbox_pixels=detection.bbox,
        )
        class_ids.append(int(detection.class_id))
        features.append(
            detection_attention_features(
                detection, image_width, image_height, predicted_depth
            )
        )
    return (
        np.asarray(class_ids, dtype=np.int64),
        np.asarray(features, dtype=np.float32).reshape(
            -1, OBJECT_CONTINUOUS_FEATURE_DIM
        ),
    )


def padded_object_condition_from_detections(
    detections: Sequence[ObjectDetection],
    image_width: int,
    image_height: int,
    regressor,
    max_objects: int,
    rgb_np: Optional[np.ndarray] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_rows, feature_rows = object_rows_from_detections(
        detections, image_width, image_height, regressor, rgb_np=rgb_np
    )
    class_ids, features, mask = pad_object_rows(
        class_rows, feature_rows, max_objects
    )
    return (
        torch.from_numpy(class_ids).unsqueeze(0),
        torch.from_numpy(features).unsqueeze(0),
        torch.from_numpy(mask).unsqueeze(0),
    )


@dataclass
class ObjectAttentionCache:
    image_keys: np.ndarray
    offsets: np.ndarray
    class_ids: np.ndarray
    features: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        normalized = [
            normalize_image_key(key) for key in self.image_keys.tolist()
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Object cache contains duplicate image keys")
        self._index = {key: idx for idx, key in enumerate(normalized)}
        self._validate()

    def _validate(self) -> None:
        if (
            self.features.ndim != 2
            or self.features.shape[1] != OBJECT_CONTINUOUS_FEATURE_DIM
        ):
            raise ValueError(
                f"Unsupported object feature shape {self.features.shape}; "
                f"expected [N,{OBJECT_CONTINUOUS_FEATURE_DIM}]"
            )
        if len(self.offsets) != len(self.image_keys) + 1:
            raise ValueError("Object cache offsets do not match image keys")
        if (
            len(self.offsets) == 0
            or self.offsets[0] != 0
            or np.any(np.diff(self.offsets) < 0)
            or self.offsets[-1] != len(self.class_ids)
            or len(self.class_ids) != len(self.features)
        ):
            raise ValueError("Object cache offsets or row counts are invalid")
        if not np.isfinite(self.features).all():
            raise ValueError("Object cache contains non-finite features")

    @classmethod
    def load(cls, path: str | Path) -> "ObjectAttentionCache":
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            return cls(
                image_keys=payload["image_keys"].copy(),
                offsets=payload["offsets"].astype(np.int64),
                class_ids=payload["class_ids"].astype(np.int64),
                features=payload["features"].astype(np.float32),
                metadata=metadata,
            )

    def contains(self, image_path: str | Path) -> bool:
        return normalize_image_key(image_path) in self._index

    def padded(
        self, image_path: str | Path, max_objects: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self._index.get(normalize_image_key(image_path))
        if idx is None:
            return pad_object_rows(
                np.empty(0, dtype=np.int64),
                np.empty((0, OBJECT_CONTINUOUS_FEATURE_DIM), dtype=np.float32),
                max_objects,
            )
        start, stop = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return pad_object_rows(
            self.class_ids[start:stop], self.features[start:stop], max_objects
        )

    def _padded_at_index(
        self, idx: int, max_objects: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start, stop = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return pad_object_rows(
            self.class_ids[start:stop], self.features[start:stop], max_objects
        )

    def resolve_index(
        self,
        image_path: str | Path,
        *,
        rgb_root: str | Path | None = None,
    ) -> int | None:
        rel = str(image_path).replace("\\", "/")
        candidates = [normalize_image_key(image_path)]
        if rgb_root is not None:
            candidates.append(normalize_image_key(Path(rgb_root) / rel))
        for key in candidates:
            if key in self._index:
                return self._index[key]

        rel_key = normalize_image_key(rel)
        matches = [
            key
            for key in self._index
            if key == rel_key or key.endswith("/" + rel_key)
        ]
        if len(matches) == 1:
            return self._index[matches[0]]
        if len(matches) > 1:
            basename = rel_key.rsplit("/", 1)[-1]
            basename_matches = [key for key in matches if key.endswith("/" + basename)]
            if len(basename_matches) == 1:
                return self._index[basename_matches[0]]
        return None

    def padded_for_eval(
        self,
        image_path: str | Path,
        max_objects: int,
        *,
        rgb_root: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self.resolve_index(image_path, rgb_root=rgb_root)
        if idx is None:
            return pad_object_rows(
                np.empty(0, dtype=np.int64),
                np.empty((0, OBJECT_CONTINUOUS_FEATURE_DIM), dtype=np.float32),
                max_objects,
            )
        return self._padded_at_index(idx, max_objects)

    def object_count(
        self,
        image_path: str | Path,
        *,
        rgb_root: str | Path | None = None,
    ) -> int:
        idx = self.resolve_index(image_path, rgb_root=rgb_root)
        if idx is None:
            return 0
        return int(self.offsets[idx + 1] - self.offsets[idx])


class ObjectAttentionEncoder(ModelMixin, ConfigMixin):
    """Project a padded set of object descriptors to UNet cross-attention tokens."""

    @register_to_config
    def __init__(
        self,
        num_classes: int = 91,
        class_embed_dim: int = 64,
        continuous_dim: int = OBJECT_CONTINUOUS_FEATURE_DIM,
        hidden_dim: int = 512,
        num_hidden_layers: int = 3,
        cross_attention_dim: int = 1024,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.continuous_norm = nn.LayerNorm(continuous_dim)
        in_dim = class_embed_dim + continuous_dim
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_hidden_layers - 1)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, cross_attention_dim)
        self.out_norm = nn.LayerNorm(cross_attention_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        class_ids: torch.Tensor,
        continuous_features: torch.Tensor,
        object_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        class_emb = self.class_embedding(
            class_ids.clamp(0, self.class_embedding.num_embeddings - 1)
        )
        continuous = self.continuous_norm(continuous_features)
        hidden = self.stem(torch.cat([class_emb, continuous], dim=-1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        tokens = self.out_norm(self.out_proj(hidden))
        if object_mask is not None:
            tokens = tokens * object_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens


def apply_object_condition_dropout(
    object_mask: torch.Tensor,
    dropout_p: float,
    *,
    training: bool,
) -> torch.Tensor:
    if not training or dropout_p <= 0:
        return object_mask
    if dropout_p >= 1:
        return torch.zeros_like(object_mask)
    keep = torch.rand(
        object_mask.shape[0], 1, device=object_mask.device
    ) >= dropout_p
    return object_mask & keep


def append_object_attention_tokens(
    text_embeddings: torch.Tensor,
    object_tokens: torch.Tensor | None,
    object_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Append real object tokens and return the matching encoder attention mask."""
    if object_tokens is None or object_mask is None or not bool(object_mask.any()):
        return text_embeddings, None
    if object_tokens.shape[:2] != object_mask.shape:
        raise ValueError(
            f"object token/mask shape mismatch: {object_tokens.shape} vs {object_mask.shape}"
        )
    text_mask = torch.ones(
        text_embeddings.shape[:2],
        device=text_embeddings.device,
        dtype=object_mask.dtype,
    )
    states = torch.cat(
        [text_embeddings, object_tokens.to(text_embeddings.dtype)], dim=1
    )
    return states, torch.cat([text_mask, object_mask], dim=1)


def encode_object_attention_condition(
    text_embeddings: torch.Tensor,
    encoder: ObjectAttentionEncoder,
    class_ids: torch.Tensor,
    continuous_features: torch.Tensor,
    object_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Move, encode, and append one padded object condition batch."""
    device = text_embeddings.device
    encoder_dtype = next(encoder.parameters()).dtype
    mask = object_mask.to(device=device, dtype=torch.bool)
    tokens = encoder(
        class_ids.to(device=device, dtype=torch.long),
        continuous_features.to(device=device, dtype=encoder_dtype),
        mask,
    )
    return append_object_attention_tokens(text_embeddings, tokens, mask)


def stack_object_rows(
    rows: Sequence[tuple[str, Sequence[int], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_keys: list[str] = []
    offsets = [0]
    class_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    for image_key, class_ids, features in rows:
        image_keys.append(normalize_image_key(image_key))
        class_arr = np.asarray(class_ids, dtype=np.int64)
        feature_arr = np.asarray(features, dtype=np.float32).reshape(
            -1, OBJECT_CONTINUOUS_FEATURE_DIM
        )
        if len(class_arr) != len(feature_arr):
            raise ValueError(f"class/features length mismatch for {image_key}")
        class_parts.append(class_arr)
        feature_parts.append(feature_arr)
        offsets.append(offsets[-1] + len(class_arr))
    all_classes = (
        np.concatenate(class_parts) if class_parts else np.empty(0, dtype=np.int64)
    )
    all_features = (
        np.concatenate(feature_parts)
        if feature_parts
        else np.empty((0, OBJECT_CONTINUOUS_FEATURE_DIM), dtype=np.float32)
    )
    return (
        np.asarray(image_keys),
        np.asarray(offsets, dtype=np.int64),
        all_classes,
        all_features,
    )
