"""Supervision: which contour pixels sit on a real depth discontinuity.

Generated from GT depth, so no hand annotation is needed - the nearest prior work
(Ramamonjisoa et al., CVPR 2020) annotated 654 frames of occlusion boundaries by hand.
GT is used here and nowhere else: it never enters the inference path.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .paths import cache_path  # noqa: F401  (keeps the import surface in one place)


def contour_labels(gt: np.ndarray, valid: np.ndarray, t: float,
                   label_px: float) -> np.ndarray:
    """True within `label_px` of a depth step of more than `t` percent.

    Computed over the whole frame, including when training crops: a contour just outside
    a crop still decides the labels at its edge.
    """
    from eval_mask_contour_localization import discontinuities
    dist = ndi.distance_transform_edt(~discontinuities(gt, valid, t))
    return dist <= label_px
