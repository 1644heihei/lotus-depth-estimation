"""Dataset loader for Approach-A offline detail training artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.hypersim_dataset import HypersimImageDepthNormalTransform, get_hypersim_dataset_depth_normal
from utils.object_condition import class_map_path, class_map_to_tensor
from utils.object_detection_cache import detections_to_mask, load_detections
from utils.object_attention_condition import ObjectAttentionCache


def count_detections_at_thr(rgb_path: str | Path, detail_root: str | Path, score_thr: float) -> int:
    detections = load_detections(rgb_path, detail_root)
    return sum(1 for d in detections if d.score >= score_thr)


def load_detail_train_manifest(manifest_path: str | Path) -> Tuple[List[str], List[str], dict]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not entries:
        raise ValueError(f"Manifest has no entries: {manifest_path}")
    rgb_paths = [e["rgb_path"] for e in entries]
    depth_paths = [e.get("depth_path") or _guess_depth_path_from_rgb(e["rgb_path"]) for e in entries]
    return rgb_paths, depth_paths, payload


def _guess_depth_path_from_rgb(rgb_path: str) -> str:
    p = Path(rgb_path)
    if p.name.startswith("rgb_cam_"):
        depth_name = p.name.replace("rgb_cam_", "depth_plane_cam_")
        candidate = p.parent / depth_name
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Could not infer depth path for {rgb_path}")


def filter_paths_with_detections(
    rgb_paths: List[str],
    depth_paths: List[str],
    detail_root: str | Path,
    *,
    detection_score_thr: float,
) -> Tuple[List[str], List[str]]:
    kept_rgb: List[str] = []
    kept_depth: List[str] = []
    for rgb, depth in zip(rgb_paths, depth_paths):
        if count_detections_at_thr(rgb, detail_root, detection_score_thr) > 0:
            kept_rgb.append(rgb)
            kept_depth.append(depth)
    return kept_rgb, kept_depth
from utils.object_pre_depth import load_pre_depth_artifacts, pre_depth_path, valid_mask_path
from utils.object_size_condition import rasterize_class_and_size_maps, size_map_to_tensor

logger = logging.getLogger(__name__)


def parse_hf_uri(uri: str) -> Tuple[str, str]:
    without = uri[len("hf://") :]
    if "?" in without:
        without = without.split("?", 1)[0]
    parts = without.split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid HF uri: {uri}")
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def open_rgb_depth(rgb_path: str, depth_path: str):
    if str(rgb_path).startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo_id, rgb_file = parse_hf_uri(rgb_path)
        _, depth_file = parse_hf_uri(depth_path)
        local_rgb = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rgb_file)
        local_depth = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=depth_file)
        image = Image.open(local_rgb).convert("RGB")
        depth = np.array(Image.open(local_depth))
    else:
        image = Image.open(rgb_path).convert("RGB")
        depth = np.array(Image.open(depth_path))
    if depth.dtype == np.uint16 and depth.max() > 1000:
        depth = depth.astype(np.float32) / 100.0
    else:
        depth = depth.astype(np.float32)
    depth = np.clip(depth, 1e-4, None)
    return image, depth


def list_rgb_images(rgb_root: str | Path, pattern: str = "rgb_cam_*.png") -> List[Path]:
    rgb_root = Path(rgb_root)
    return sorted(rgb_root.rglob(pattern))


def artifacts_ready(
    rgb_path: str | Path,
    detail_root: str | Path,
    pre_depth_root: str | Path | None = None,
) -> bool:
    rgb_path = Path(rgb_path)
    detail_root = Path(detail_root)
    pre_depth_root = Path(pre_depth_root) if pre_depth_root is not None else detail_root
    return (
        pre_depth_path(rgb_path, pre_depth_root).is_file()
        and valid_mask_path(rgb_path, pre_depth_root).is_file()
        and class_map_path(rgb_path, detail_root).is_file()
    )


class DetailTrainDataset(torch.utils.data.Dataset):
    """Reads RGB/GT depth plus pre-built pre-depth, valid mask, and class map."""

    def __init__(
        self,
        rgb_root: str | Path,
        detail_root: str | Path,
        transform: HypersimImageDepthNormalTransform,
        *,
        depth_paths: Optional[List[str]] = None,
        rgb_paths: Optional[List[str]] = None,
        require_artifacts: bool = True,
        detection_score_thr: float = 0.5,
        require_detections: bool = False,
        manifest_path: Optional[str] = None,
        pre_depth_root: Optional[str | Path] = None,
        object_attention_cache_path: Optional[str | Path] = None,
        max_objects: int = 16,
    ):
        self.rgb_root = Path(rgb_root)
        self.detail_root = Path(detail_root)
        self.pre_depth_root = Path(pre_depth_root) if pre_depth_root is not None else self.detail_root
        self.transform = transform
        self.detection_score_thr = float(detection_score_thr)
        self.object_attention_cache = (
            ObjectAttentionCache.load(object_attention_cache_path)
            if object_attention_cache_path is not None
            else None
        )
        self.max_objects = int(max_objects)
        if self.object_attention_cache is not None:
            cache_threshold = self.object_attention_cache.metadata.get(
                "detection_score_thr"
            )
            if (
                cache_threshold is not None
                and abs(float(cache_threshold) - self.detection_score_thr) > 1e-8
            ):
                raise ValueError(
                    "Object attention cache threshold does not match "
                    f"dataset threshold: {cache_threshold} vs "
                    f"{self.detection_score_thr}"
                )

        if manifest_path is not None:
            rgb_paths, depth_paths, _ = load_detail_train_manifest(manifest_path)
            self.rgb_paths = list(rgb_paths)
            self.depth_paths = list(depth_paths)
        elif rgb_paths is None:
            rgb_paths = [str(p) for p in list_rgb_images(self.rgb_root)]
            self.rgb_paths = list(rgb_paths)
            self.depth_paths = depth_paths or [self._guess_depth_path(p) for p in self.rgb_paths]
        else:
            self.rgb_paths = list(rgb_paths)
            self.depth_paths = depth_paths or [self._guess_depth_path(p) for p in self.rgb_paths]

        if self.object_attention_cache is not None:
            pairs = [
                (rgb, depth)
                for rgb, depth in zip(self.rgb_paths, self.depth_paths)
                if self.object_attention_cache.contains(rgb)
            ]
            if not pairs:
                raise FileNotFoundError(
                    "Object attention cache has no entries for this dataset."
                )
            self.rgb_paths, self.depth_paths = map(list, zip(*pairs))

        if require_artifacts:
            pairs = [
                (rgb, depth)
                for rgb, depth in zip(self.rgb_paths, self.depth_paths)
                if artifacts_ready(rgb, self.detail_root, self.pre_depth_root)
            ]
            if not pairs:
                raise FileNotFoundError(
                    f"No complete detail artifacts found under {self.detail_root}. "
                    "Run utils/build_detail_train_dataset.py first."
                )
            self.rgb_paths, self.depth_paths = zip(*pairs)
            self.rgb_paths = list(self.rgb_paths)
            self.depth_paths = list(self.depth_paths)

        if require_detections:
            before = len(self.rgb_paths)
            self.rgb_paths, self.depth_paths = filter_paths_with_detections(
                self.rgb_paths,
                self.depth_paths,
                self.detail_root,
                detection_score_thr=self.detection_score_thr,
            )
            if not self.rgb_paths:
                raise FileNotFoundError(
                    f"No images with detections (score>={self.detection_score_thr}) under {self.detail_root}."
                )
            logger.info(
                "Filtered to images with detections: %d -> %d (score>=%.2f)",
                before,
                len(self.rgb_paths),
                self.detection_score_thr,
            )

    def _guess_depth_path(self, rgb_path: str) -> str:
        p = Path(rgb_path)
        if p.name.startswith("rgb_cam_"):
            depth_name = p.name.replace("rgb_cam_", "depth_plane_cam_")
            candidate = p.parent / depth_name
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"Could not infer depth path for {rgb_path}")

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def _load_condition_maps(
        self, rgb_path: str, target_hw: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.object_attention_cache is not None:
            zeros = np.zeros(target_hw, dtype=np.float32)
            return zeros, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

        pre_depth, valid_mask = load_pre_depth_artifacts(
            rgb_path, self.pre_depth_root
        )
        if pre_depth is None or valid_mask is None:
            raise FileNotFoundError(f"Missing pre-depth artifacts for {rgb_path}")

        detections = load_detections(rgb_path, self.detail_root)
        if self.detection_score_thr > 0:
            detections = [
                detection
                for detection in detections
                if detection.score >= self.detection_score_thr
            ]
            keep = detections_to_mask(
                detections, valid_mask.shape[0], valid_mask.shape[1]
            )
            valid_mask = valid_mask.astype(np.float32) * keep

        class_map, size_w, size_h = rasterize_class_and_size_maps(
            detections, valid_mask.shape[0], valid_mask.shape[1]
        )
        if (
            self.detection_score_thr <= 0
            and not detections
            and class_map_path(rgb_path, self.detail_root).is_file()
        ):
            class_map = np.load(class_map_path(rgb_path, self.detail_root))
        return pre_depth, valid_mask, class_map, size_w, size_h

    def __getitem__(self, idx: int):
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]

        image, depth = open_rgb_depth(rgb_path, depth_path)

        h, w = depth.shape[:2]
        fallback_normal = np.zeros((h, w, 3), dtype=np.float32)
        fallback_normal[..., 2] = 1.0

        pixel_values, depth_values, normal_values = self.transform(image, depth, fallback_normal)

        pre_depth, valid_mask, class_map, size_w, size_h = (
            self._load_condition_maps(rgb_path, pixel_values.shape[-2:])
        )

        pre_depth_t = self._resize_map(pre_depth, pixel_values.shape[-2:])
        valid_mask_t = self._resize_map(valid_mask, pixel_values.shape[-2:])
        class_map_t = self._resize_map(class_map.astype(np.float32), pixel_values.shape[-2:], nearest=True)
        size_w_t = self._resize_map(size_w, pixel_values.shape[-2:], nearest=True)
        size_h_t = self._resize_map(size_h, pixel_values.shape[-2:], nearest=True)
        class_map_norm = torch.from_numpy(class_map_to_tensor(class_map_t.numpy())).unsqueeze(0)
        size_w_norm = torch.from_numpy(size_map_to_tensor(size_w_t.numpy())).unsqueeze(0)
        size_h_norm = torch.from_numpy(size_map_to_tensor(size_h_t.numpy())).unsqueeze(0)

        output = {
            "pixel_values": pixel_values,
            "depth_values": depth_values,
            "normal_values": normal_values,
            "pre_depth_values": pre_depth_t.unsqueeze(0).repeat(3, 1, 1),
            "pre_depth_valid_mask": valid_mask_t.unsqueeze(0),
            "class_map_values": class_map_norm,
            "size_w_values": size_w_norm,
            "size_h_values": size_h_norm,
            "image_path": rgb_path,
            "depth_path": depth_path,
        }
        if self.object_attention_cache is not None:
            object_class_ids, object_features, object_mask = (
                self.object_attention_cache.padded(rgb_path, self.max_objects)
            )
            output.update(
                {
                    "object_class_ids": torch.from_numpy(object_class_ids),
                    "object_features": torch.from_numpy(object_features),
                    "object_mask": torch.from_numpy(object_mask),
                }
            )
        return output

    def _resize_map(self, arr: np.ndarray, size_hw: Tuple[int, int], nearest: bool = False) -> torch.Tensor:
        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mode = "nearest" if nearest else "bilinear"
        if mode == "bilinear":
            t = F.interpolate(t, size=size_hw, mode=mode, align_corners=False)
        else:
            t = F.interpolate(t, size=size_hw, mode=mode)
        return t.squeeze(0).squeeze(0)


def get_detail_train_dataset(
    rgb_root: str,
    detail_root: str,
    resolution: int,
    random_flip: bool,
    norm_type: str,
    truncnorm_min: float = 0.02,
    align_cam_normal: bool = False,
    detection_score_thr: float = 0.5,
    require_detections: bool = False,
    manifest_path: Optional[str] = None,
    pre_depth_root: Optional[str] = None,
    object_attention_cache_path: Optional[str] = None,
    max_objects: int = 16,
):
    """Build detail-train dataset using Hypersim index when possible."""
    manifest_rgb_paths: Optional[List[str]] = None
    manifest_depth_paths: Optional[List[str]] = None
    if manifest_path is not None:
        manifest_rgb_paths, manifest_depth_paths, _ = load_detail_train_manifest(manifest_path)
        rgb_paths = manifest_rgb_paths
        depth_paths = manifest_depth_paths
    elif rgb_root.startswith("hf://"):
        base_dataset, _, _ = get_hypersim_dataset_depth_normal(
            rgb_root,
            resolution,
            random_flip,
            norm_type=norm_type,
            truncnorm_min=truncnorm_min,
            align_cam_normal=align_cam_normal,
        )
        rgb_paths = base_dataset["image"]
        depth_paths = base_dataset["depth"]
        sample_path = rgb_paths[0]
        if str(sample_path).startswith("hf://"):
            from utils.hypersim_hf_index import collect_hypersim_rgb_depth_pairs

            uri = rgb_root[len("hf://") :]
            if "?" in uri:
                uri = uri.split("?", 1)[0]
            parts = uri.split("/")
            repo_id = f"{parts[0]}/{parts[1]}"
            split = parts[2] if len(parts) > 2 else "train"
            pairs = collect_hypersim_rgb_depth_pairs(repo_id=repo_id, split=split)
            rgb_paths = [f"hf://{repo_id}/{rgb}" for rgb, _ in pairs]
            depth_paths = [f"hf://{repo_id}/{depth}" for _, depth in pairs]
    else:
        rgb_paths = [str(p) for p in list_rgb_images(rgb_root)]
        depth_paths = []
        for rgb in rgb_paths:
            p = Path(rgb)
            depth_name = p.name.replace("rgb_cam_", "depth_plane_cam_")
            depth_paths.append(str(p.parent / depth_name))

    # Resolve transform size from first local rgb.
    local_rgb = next((p for p in rgb_paths if not str(p).startswith("hf://")), None)
    if local_rgb is None:
        new_h = new_w = resolution
    else:
        w, h = Image.open(local_rgb).size
        if h > w:
            new_w = resolution
            new_h = int(resolution * h / w)
        else:
            new_h = resolution
            new_w = int(resolution * w / h)

    transform = HypersimImageDepthNormalTransform(
        (new_h, new_w), random_flip, norm_type, truncnorm_min, align_cam_normal
    )
    dataset = DetailTrainDataset(
        rgb_root=rgb_root,
        detail_root=detail_root,
        transform=transform,
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        require_artifacts=object_attention_cache_path is None,
        detection_score_thr=detection_score_thr,
        require_detections=require_detections,
        manifest_path=manifest_path,
        pre_depth_root=pre_depth_root,
        object_attention_cache_path=object_attention_cache_path,
        max_objects=max_objects,
    )

    def preprocess_noop(examples):
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        depth_values = torch.stack([e["depth_values"] for e in examples])
        normal_values = torch.stack([e["normal_values"] for e in examples])
        pre_depth_values = torch.stack([e["pre_depth_values"] for e in examples])
        pre_depth_valid_mask = torch.stack([e["pre_depth_valid_mask"] for e in examples])
        class_map_values = torch.stack([e["class_map_values"] for e in examples])
        size_w_values = torch.stack([e["size_w_values"] for e in examples])
        size_h_values = torch.stack([e["size_h_values"] for e in examples])
        batch = {
            "pixel_values": pixel_values.float(),
            "depth_values": depth_values.float(),
            "normal_values": normal_values.float(),
            "pre_depth_values": pre_depth_values.float(),
            "pre_depth_valid_mask": pre_depth_valid_mask.float(),
            "class_map_values": class_map_values.float(),
            "size_w_values": size_w_values.float(),
            "size_h_values": size_h_values.float(),
            "image_pathes": [e["image_path"] for e in examples],
            "depth_paths": [e["depth_path"] for e in examples],
        }
        if "object_class_ids" in examples[0]:
            batch.update(
                {
                    "object_class_ids": torch.stack(
                        [e["object_class_ids"] for e in examples]
                    ).long(),
                    "object_features": torch.stack(
                        [e["object_features"] for e in examples]
                    ).float(),
                    "object_mask": torch.stack(
                        [e["object_mask"] for e in examples]
                    ).bool(),
                }
            )
        return batch

    return dataset, preprocess_noop, collate_fn
