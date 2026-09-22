"""Fit a prediction to GT in the space the model actually predicts in.

Lotus emits affine-invariant disparity, so `align_to_gt` in
eval_object_oracle_ceiling.py converts GT to disparity, fits scale and shift
there, and filters on `pred > 0`. Marigold V2's Log-stage2 emits affine-invariant
LOG depth over [-1, 1]: that filter would drop roughly every pixel nearer than
the affine zero crossing, and the fit would be taken in the wrong space entirely.

Both spaces are a two-parameter fit followed by an inverse transform, so the only
difference is which transform and which validity test. This module holds the pair
so the scripts can switch with a flag instead of growing a second aligner each.

The log conversions mirror marigold-v2/evaluation/src/util/alignment.py
(depth2log_space / log_space2depth) so the numbers stay comparable to the ones
Marigold V2 reports; align_depth_least_square is already the same function in both
repositories, both descended from Marigold V1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.util.alignment import (  # noqa: E402
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)

SPACES = ("disparity", "log")
LOG_EPS = 1e-6
DEPTH_MIN, DEPTH_MAX = 1e-3, 10.0


def depth2log_space(depth: np.ndarray, eps: float = LOG_EPS):
    """log(depth + eps) where depth is positive, 0 elsewhere, plus that mask."""
    out = np.zeros_like(depth)
    positive = depth > 0
    out[positive] = np.log(depth[positive] + eps)
    return out, positive


def log_space2depth(log_space: np.ndarray, eps: float = LOG_EPS) -> np.ndarray:
    return np.exp(log_space) - eps


def align_to_gt_disparity(pred, gt, valid, *, min_px: int = 100):
    """Lotus: prediction is affine-invariant disparity, strictly positive."""
    gt_disp, gt_ok = depth2disparity(depth=gt, return_mask=True)
    # `pred > 0` is meaningful here: a non-positive disparity is not a distance.
    valid_nn = valid & gt_ok & (pred > 0)
    if valid_nn.sum() < min_px:
        return None
    aligned, _, _ = align_depth_least_square(
        gt_arr=gt_disp, pred_arr=pred, valid_mask_arr=valid_nn, return_scale_shift=True
    )
    return np.clip(disparity2depth(np.clip(aligned, DEPTH_MIN, None)), DEPTH_MIN, DEPTH_MAX)


def align_to_gt_log(pred, gt, valid, *, min_px: int = 100):
    """Marigold V2: prediction is affine-invariant log depth and may be negative.

    No `pred > 0` test - the sign of a log-depth prediction carries no validity
    information, only its position along an unknown affine ramp.
    """
    gt_log, gt_ok = depth2log_space(gt)
    valid_nn = valid & gt_ok
    if valid_nn.sum() < min_px:
        return None
    aligned, _, _ = align_depth_least_square(
        gt_arr=gt_log, pred_arr=pred, valid_mask_arr=valid_nn, return_scale_shift=True
    )
    return np.clip(log_space2depth(aligned), DEPTH_MIN, DEPTH_MAX)


def get_aligner(space: str):
    """Return the aligner for `space`; raises rather than silently defaulting."""
    if space not in SPACES:
        raise ValueError(f"unknown align space {space!r}; expected one of {SPACES}")
    return align_to_gt_disparity if space == "disparity" else align_to_gt_log


def add_align_space_arg(parser) -> None:
    """Add --align_space, defaulting to the behaviour every existing result used."""
    parser.add_argument(
        "--align_space",
        choices=SPACES,
        default="disparity",
        help="Space the prediction lives in: 'disparity' for Lotus (default, what "
             "every published result here used), 'log' for Marigold V2 Log-stage2.",
    )
