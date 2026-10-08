"""The contour selector: one definition of what the network eats and what it keeps.

Training and inference used to build the five input channels independently, in
train_contour_selector.ContourDataset and apply_contour_selector.main. Two of those steps
are silent if they disagree - the selector still runs, still produces a mask, and only the
BF1 at the end is wrong:

  - the per-image standardisation (the prediction is affine-invariant, so its level and
    scale carry no information, and a mismatch feeds the network magnitudes it never saw);
  - the prediction space (Lotus emits disparity, which rises as depth falls, so handing it
    over raw inverts the signal).

So they live here once, in `inputs`, and both callers import them. `selection` holds the
top-share rule, `labels` the supervision, `model` the network.
"""

from __future__ import annotations

from .inputs import (PRED_SPACES, build_input, in_channels, load_candidates,
                     load_prediction, standardise, to_log_depth)
from .labels import contour_labels
from .model import UNet
from .paths import cache_path
from .selection import keep_count, top_share_mask, top_share_mask_torch

__all__ = [
    "PRED_SPACES", "build_input", "in_channels", "load_candidates", "load_prediction",
    "standardise", "to_log_depth", "contour_labels", "UNet", "cache_path",
    "top_share_mask", "top_share_mask_torch", "keep_count",
]
