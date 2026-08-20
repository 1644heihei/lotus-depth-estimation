"""ROI feature extractor using a frozen CNN backbone (ResNet-18)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


def crop_bbox_square_padded(
    image: np.ndarray,
    bbox: Sequence[Union[int, float]],
    target_size: int = 224,
) -> np.ndarray:
    """Crop bbox with square aspect ratio padding and resize to target_size.

    Args:
        image: HxWxC uint8 numpy array.
        bbox: [x1, y1, x2, y2] (pixel coordinates or normalized if floats < 1).
        target_size: Output resolution (square).

    Returns:
        target_size x target_size x C uint8 numpy array.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if max(x1, x2) <= 1.0 and max(y1, y2) <= 1.0:
        x1 = x1 * w
        x2 = x2 * w
        y1 = y1 * h
        y2 = y2 * h

    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    bw = x2 - x1
    bh = y2 - y1
    side = max(bw, bh)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    sq_x1 = int(round(cx - side / 2.0))
    sq_y1 = int(round(cy - side / 2.0))
    sq_x2 = sq_x1 + side
    sq_y2 = sq_y1 + side

    # Pad if outside image bounds
    pad_left = max(0, -sq_x1)
    pad_top = max(0, -sq_y1)
    pad_right = max(0, sq_x2 - w)
    pad_bottom = max(0, sq_y2 - h)

    crop_x1 = max(0, sq_x1)
    crop_y1 = max(0, sq_y1)
    crop_x2 = min(w, sq_x2)
    crop_y2 = min(h, sq_y2)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        c = image.shape[2] if image.ndim == 3 else 1
        padded = np.zeros((side, side, c) if image.ndim == 3 else (side, side), dtype=image.dtype)
        padded[pad_top : pad_top + (crop_y2 - crop_y1), pad_left : pad_left + (crop_x2 - crop_x1)] = crop
        crop = padded

    pil_img = Image.fromarray(crop)
    resized = pil_img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(resized)


class RoiFeatureExtractor:
    """Extract visual features from bounding box crops using frozen ResNet-18."""

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
        dtype: torch.dtype = torch.float32,
        weights: Optional[ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype

        model = resnet18(weights=weights)
        # Keep everything up to avgpool; output shape [B, 512, 1, 1]
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.backbone.to(self.device, dtype=self.dtype)

        self.feature_dim = 512
        self.target_size = 224

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def extract_single(
        self,
        image_np: np.ndarray,
        bbox: Sequence[Union[int, float]],
    ) -> np.ndarray:
        """Extract 512-dim feature for a single bbox."""
        crop = crop_bbox_square_padded(image_np, bbox, target_size=self.target_size)
        if crop.ndim == 2:
            crop = np.stack([crop] * 3, axis=-1)
        elif crop.shape[2] == 4:
            crop = crop[:, :, :3]
        tensor = self.transform(crop).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        feat = self.backbone(tensor).flatten(1)  # [1, 512]
        return feat.squeeze(0).cpu().to(torch.float32).numpy()

    @torch.no_grad()
    def extract_batch(
        self,
        image_np: np.ndarray,
        bboxes: Sequence[Sequence[Union[int, float]]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract 512-dim features for multiple bboxes in an image."""
        if not bboxes:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        crops = []
        for bbox in bboxes:
            crop = crop_bbox_square_padded(image_np, bbox, target_size=self.target_size)
            if crop.ndim == 2:
                crop = np.stack([crop] * 3, axis=-1)
            elif crop.shape[2] == 4:
                crop = crop[:, :, :3]
            crops.append(self.transform(crop))

        features_list = []
        for i in range(0, len(crops), batch_size):
            batch_tensors = torch.stack(crops[i : i + batch_size]).to(
                device=self.device, dtype=self.dtype
            )
            feats = self.backbone(batch_tensors).flatten(1)  # [B, 512]
            features_list.append(feats.cpu().to(torch.float32).numpy())

        return np.concatenate(features_list, axis=0)

    @torch.no_grad()
    def extract_crop_tensors(
        self,
        crop_tensors: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features directly from preprocessed crop tensors [B, 3, 224, 224]."""
        crop_tensors = crop_tensors.to(device=self.device, dtype=self.dtype)
        return self.backbone(crop_tensors).flatten(1)


class RoiFeatureCache:
    """Indexed cache of precomputed per-object ROI features."""

    def __init__(
        self,
        image_keys: np.ndarray,
        offsets: np.ndarray,
        roi_features: np.ndarray,
        metadata: dict,
    ):
        self.image_keys = np.asarray(image_keys)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.roi_features = np.asarray(roi_features, dtype=np.float32)
        self.metadata = metadata
        # Primary index: exact normalized key -> array index
        self._index = {str(k).replace("\\", "/"): i for i, k in enumerate(self.image_keys)}
        # Suffix index: every suffix component -> list of (key, idx) for fuzzy matching
        self._suffix_index: dict[str, list[tuple[str, int]]] = {}
        for key, idx in self._index.items():
            parts = key.split("/")
            for n in range(1, len(parts) + 1):
                suffix = "/".join(parts[-n:])
                self._suffix_index.setdefault(suffix, []).append((key, idx))

    @classmethod
    def load(cls, npz_path: Union[str, Path]) -> "RoiFeatureCache":
        import json

        npz_path = Path(npz_path)
        with np.load(npz_path, allow_pickle=False) as data:
            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"].item()))
            return cls(
                image_keys=data["image_keys"].copy(),
                offsets=data["offsets"].copy(),
                roi_features=data["roi_features"].copy(),
                metadata=metadata,
            )

    def get_features(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        key = str(image_path).replace("\\", "/")
        # Exact match first
        idx = self._index.get(key)
        if idx is None:
            # Fuzzy suffix match: try progressively shorter suffix components
            parts = key.split("/")
            for n in range(1, len(parts) + 1):
                suffix = "/".join(parts[-n:])
                candidates = self._suffix_index.get(suffix)
                if candidates and len(candidates) == 1:
                    idx = candidates[0][1]
                    break
        if idx is None:
            return None
        start, stop = self.offsets[idx], self.offsets[idx + 1]
        return self.roi_features[start:stop]

