import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


_MASK_EXTS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".npy")


def mask_relpath_for_image(image_path: str, ext: str = ".png") -> Path:
    """Map an RGB path to a mask relative path under mask_root."""
    p = Path(image_path)
    for split in ("train", "val", "test"):
        if split in p.parts:
            idx = p.parts.index(split)
            return Path(*p.parts[idx:]).with_suffix(ext)
    return Path(p.parent.name) / f"{p.stem}{ext}"


def mask_path_for_image(image_path: str, mask_root: str, ext: str = ".png") -> str:
    return str(Path(mask_root) / mask_relpath_for_image(image_path, ext=ext))


def resolve_mask_path(image_path: str, mask_root: str) -> Optional[str]:
    """Resolve mask file: prefer mirrored train/scene layout, then flat basename."""
    for ext in _MASK_EXTS:
        mirrored = mask_path_for_image(image_path, mask_root, ext=ext)
        if os.path.exists(mirrored):
            return mirrored
    stem = Path(image_path).stem
    for ext in _MASK_EXTS:
        candidate = os.path.join(mask_root, f"{stem}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def load_mask(mask_path: str) -> np.ndarray:
    """Load mask and return float map in [0, 1], shape [H, W]."""
    suffix = Path(mask_path).suffix.lower()
    if suffix == ".npy":
        mask = np.load(mask_path)
    else:
        mask = np.array(Image.open(mask_path).convert("L"))
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return np.clip(mask, 0.0, 1.0)


def load_mask_for_image(image_path: str, mask_root: Optional[str]) -> Optional[np.ndarray]:
    if not mask_root:
        return None
    path = resolve_mask_path(image_path, mask_root)
    if not path:
        return None
    return load_mask(path)


def mask_to_tensor(mask: Optional[np.ndarray], device: torch.device, batch_size: int = 1) -> Optional[torch.Tensor]:
    """Convert HxW mask to tensor [B, 1, H, W]."""
    if mask is None:
        return None
    t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    if batch_size > 1:
        t = t.repeat(batch_size, 1, 1, 1)
    return t.to(device)


def resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


def connected_component_boxes(mask: np.ndarray, threshold: float = 0.5, min_area: int = 400) -> List[Tuple[int, int, int, int]]:
    """Extract connected-component boxes as (x1, y1, x2, y2)."""
    binary = (mask >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def expand_box(box: Tuple[int, int, int, int], image_hw: Tuple[int, int], expand_ratio: float = 0.12) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h, w = image_hw
    bw = x2 - x1
    bh = y2 - y1
    ex = int(round(bw * expand_ratio))
    ey = int(round(bh * expand_ratio))
    x1n = max(0, x1 - ex)
    y1n = max(0, y1 - ey)
    x2n = min(w, x2 + ex)
    y2n = min(h, y2 + ey)
    return x1n, y1n, x2n, y2n


def make_soft_weight(mask: np.ndarray, blur_ksize: int = 31) -> np.ndarray:
    k = max(3, blur_ksize)
    if k % 2 == 0:
        k += 1
    soft = cv2.GaussianBlur(mask.astype(np.float32), (k, k), sigmaX=0)
    return np.clip(soft, 0.0, 1.0)


def tensorize_rgb_np(rgb_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 RGB np image [H, W, 3] to normalized tensor [1,3,H,W] in [-1,1]."""
    rgb = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    rgb = rgb / 127.5 - 1.0
    return rgb.to(device)


def interpolate_tensor(t: torch.Tensor, size_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if mode in ("bilinear", "bicubic"):
        return F.interpolate(t, size=size_hw, mode=mode, align_corners=False)
    return F.interpolate(t, size=size_hw, mode=mode)
