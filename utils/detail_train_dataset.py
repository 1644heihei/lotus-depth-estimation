"""Dataset loader for Approach-A offline detail training artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.hypersim_dataset import HypersimImageDepthNormalTransform, get_hypersim_dataset_depth_normal
from utils.object_condition import class_map_path, class_map_to_tensor, rasterize_class_map
from utils.object_detection_cache import detections_to_mask, load_detections
from utils.object_pre_depth import load_pre_depth_artifacts, pre_depth_path, valid_mask_path


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


def artifacts_ready(rgb_path: str | Path, detail_root: str | Path) -> bool:
    rgb_path = Path(rgb_path)
    detail_root = Path(detail_root)
    return (
        pre_depth_path(rgb_path, detail_root).is_file()
        and valid_mask_path(rgb_path, detail_root).is_file()
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
    ):
        self.rgb_root = Path(rgb_root)
        self.detail_root = Path(detail_root)
        self.transform = transform
        self.detection_score_thr = float(detection_score_thr)

        if rgb_paths is None:
            rgb_paths = [str(p) for p in list_rgb_images(self.rgb_root)]
        self.rgb_paths = list(rgb_paths)
        self.depth_paths = depth_paths or [self._guess_depth_path(p) for p in self.rgb_paths]

        if require_artifacts:
            pairs = [
                (rgb, depth)
                for rgb, depth in zip(self.rgb_paths, self.depth_paths)
                if artifacts_ready(rgb, self.detail_root)
            ]
            if not pairs:
                raise FileNotFoundError(
                    f"No complete detail artifacts found under {self.detail_root}. "
                    "Run utils/build_detail_train_dataset.py first."
                )
            self.rgb_paths, self.depth_paths = zip(*pairs)
            self.rgb_paths = list(self.rgb_paths)
            self.depth_paths = list(self.depth_paths)

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

    def __getitem__(self, idx: int):
        rgb_path = self.rgb_paths[idx]
        depth_path = self.depth_paths[idx]

        image, depth = open_rgb_depth(rgb_path, depth_path)

        h, w = depth.shape[:2]
        fallback_normal = np.zeros((h, w, 3), dtype=np.float32)
        fallback_normal[..., 2] = 1.0

        pixel_values, depth_values, normal_values = self.transform(image, depth, fallback_normal)

        pre_depth, valid_mask = load_pre_depth_artifacts(rgb_path, self.detail_root)
        if pre_depth is None or valid_mask is None:
            raise FileNotFoundError(f"Missing pre-depth artifacts for {rgb_path}")

        # Training-time score filter: keep only high-confidence detections for
        # valid_mask / class_map. Offline pre_depth.npy stays as-is; regions that
        # fail the threshold are marked invalid so they are not used as condition.
        if self.detection_score_thr > 0:
            detections = [
                d
                for d in load_detections(rgb_path, self.detail_root)
                if d.score >= self.detection_score_thr
            ]
            h0, w0 = valid_mask.shape[:2]
            keep = detections_to_mask(detections, h0, w0)
            valid_mask = (valid_mask.astype(np.float32) * keep).astype(np.float32)
            class_map = rasterize_class_map(detections, h0, w0)
        else:
            class_map = np.load(class_map_path(rgb_path, self.detail_root))

        pre_depth_t = self._resize_map(pre_depth, pixel_values.shape[-2:])
        valid_mask_t = self._resize_map(valid_mask, pixel_values.shape[-2:])
        class_map_t = self._resize_map(class_map.astype(np.float32), pixel_values.shape[-2:], nearest=True)
        class_map_norm = torch.from_numpy(class_map_to_tensor(class_map_t.numpy())).unsqueeze(0)

        return {
            "pixel_values": pixel_values,
            "depth_values": depth_values,
            "normal_values": normal_values,
            "pre_depth_values": pre_depth_t.unsqueeze(0).repeat(3, 1, 1),
            "pre_depth_valid_mask": valid_mask_t.unsqueeze(0),
            "class_map_values": class_map_norm,
            "image_path": rgb_path,
            "depth_path": depth_path,
        }

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
):
    """Build detail-train dataset using Hypersim index when possible."""
    if rgb_root.startswith("hf://"):
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
        require_artifacts=True,
        detection_score_thr=detection_score_thr,
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
        return {
            "pixel_values": pixel_values.float(),
            "depth_values": depth_values.float(),
            "normal_values": normal_values.float(),
            "pre_depth_values": pre_depth_values.float(),
            "pre_depth_valid_mask": pre_depth_valid_mask.float(),
            "class_map_values": class_map_values.float(),
            "image_pathes": [e["image_path"] for e in examples],
            "depth_paths": [e["depth_path"] for e in examples],
        }

    return dataset, preprocess_noop, collate_fn
