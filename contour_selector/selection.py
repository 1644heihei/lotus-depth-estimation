"""Turning per-pixel scores into the set of contour pixels to sharpen.

The same rule in two backends: numpy for inference, which works a frame at a time off
cached arrays, and torch for the validation metric, which runs inside the training loop on
whatever device the batch is already on. They must agree - the validation precision is
read as a prediction of what the inference run will keep - so the arithmetic is written
out identically in both and `tests/test_contour_selector_api.py` checks they match.
"""

from __future__ import annotations

import numpy as np
import torch


def keep_count(n_candidates: int, retention: float) -> int:
    """How many pixels the top-share rule keeps. At least one wherever there is one."""
    return max(int(round(n_candidates * retention)), 1)


def top_share_mask(score: np.ndarray, candidates: np.ndarray,
                   retention: float) -> np.ndarray:
    """Keep the top `retention` share of the candidate pixels, per image.

    A share rather than a probability threshold: a fixed threshold keeps a different
    amount depending on how many contours a scene happens to have, so the budget the
    sharpening spends would drift from frame to frame. At least one pixel is kept wherever
    there is a candidate, so the selection is never empty by rounding.
    """
    n = int(candidates.sum())
    if n == 0:
        return np.zeros_like(candidates, dtype=bool)
    k = keep_count(n, retention)
    thr = np.partition(score[candidates], -k)[-k]
    return candidates & (score >= thr)


def top_share_mask_torch(score: torch.Tensor, candidates: torch.Tensor,
                         retention: float) -> torch.Tensor:
    """`top_share_mask` for one [H,W] score map already on a device."""
    n = int(candidates.sum())
    if n == 0:
        return torch.zeros_like(candidates, dtype=torch.bool)
    k = keep_count(n, retention)
    thr = torch.topk(score[candidates], k).values[-1]
    return candidates & (score >= thr)
