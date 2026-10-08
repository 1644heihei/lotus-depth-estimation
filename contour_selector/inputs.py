"""The selector's input channels, built the same way in training and in inference.

Channel order is RGB (3), prediction (1), SAM contour (1). Both ablations remove channels
rather than zeroing them, so `in_channels` is the one place that knows the width and a
checkpoint trained with an ablation cannot be loaded into the wrong-sized network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .paths import cache_path

PRED_SPACES = ("log", "disparity")


def to_log_depth(pred: np.ndarray, space: str) -> np.ndarray:
    """Put a cached prediction into log-depth, up to an affine.

    Marigold V2 already emits affine-invariant log depth. Lotus emits disparity, which
    rises as depth falls: fed raw it hands the selector an inverted signal, and -log(d) is
    log depth up to the affine that `standardise` removes anyway.
    """
    if space == "log":
        return pred
    if space == "disparity":
        return -np.log(np.clip(pred, 1e-3, None))
    raise ValueError(f"unknown prediction space {space!r}; expected one of {PRED_SPACES}")


def standardise(pred: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance over the WHOLE frame.

    The prediction is affine-invariant, so its absolute level and scale mean nothing and
    would differ between training (where a prediction could be aligned to GT) and
    inference (where it cannot). Over the whole frame because that is what inference sees:
    standardising a training crop on itself would teach a different distribution.
    """
    m, s = float(pred.mean()), float(pred.std())
    return (pred - m) / (s + 1e-6)


def load_prediction(rgb_path, rgb_dir, pred_dir, space: str = "log") -> np.ndarray:
    """The cached prediction for one frame, in log-depth and standardised."""
    raw = np.load(cache_path(Path(rgb_path), Path(rgb_dir), Path(pred_dir),
                             "_pred.npy")).astype(np.float32)
    return standardise(to_log_depth(raw, space))


def load_candidates(rgb_path, rgb_dir, mask_dir, valid: np.ndarray, *,
                    no_sam: bool = False) -> np.ndarray:
    """The pixels the selector is allowed to choose from.

    Normally the SAM contour intersected with the valid mask. Under --no_sam there is no
    contour to judge, so every valid pixel is a candidate and the network has to locate
    the boundaries itself - the setting a depth-only refiner works in.
    """
    if no_sam:
        return valid
    from eval_contour_feature_auc import load_contour
    h, w = valid.shape
    contour = load_contour(
        cache_path(Path(rgb_path), Path(rgb_dir), Path(mask_dir), "_seg.npz"), h, w)
    return contour & valid


def in_channels(*, no_rgb: bool = False, no_sam: bool = False) -> int:
    """Input width for the given ablation. Must match what `build_input` stacks."""
    return (0 if no_rgb else 3) + 1 + (0 if no_sam else 1)


def build_input(rgb_path, pred: np.ndarray, candidates: np.ndarray, *,
                no_rgb: bool = False, no_sam: bool = False) -> np.ndarray:
    """[C,H,W] float32 for one frame. `pred` is already standardised log depth."""
    planes = []
    if not no_rgb:
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        planes.append(rgb.transpose(2, 0, 1))
    planes.append(pred[None])
    if not no_sam:
        planes.append(candidates.astype(np.float32)[None] * 2.0 - 1.0)
    return np.concatenate(planes, axis=0)
