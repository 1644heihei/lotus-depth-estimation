"""Scale-shift alignment between depth (disparity) maps.

Used to fit per-crop predictions onto the global prediction before pasting,
so that all pre-depth patches share a single consistent scale
(see docs/pre_depth_improvement_plan.md, Step 1).
"""

from typing import Optional, Tuple

import numpy as np


def fit_scale_shift(
    src: np.ndarray,
    ref: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    trim_ratio: float = 0.05,
) -> Tuple[float, float]:
    """Least-squares fit ``src * scale + shift ~= ref``.

    Args:
        src: source map (e.g. crop prediction), any shape.
        ref: reference map (e.g. global prediction), same shape as src.
        valid_mask: optional boolean mask of pixels to use.
        trim_ratio: after an initial fit, this fraction of pixels with the
            largest absolute residuals is discarded and the fit is repeated
            (robustness against outliers such as mask bleeding / sky).

    Returns:
        (scale, shift). Falls back to (1.0, 0.0) if the fit is degenerate.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool).reshape(-1)
        src, ref = src[vm], ref[vm]

    finite = np.isfinite(src) & np.isfinite(ref)
    src, ref = src[finite], ref[finite]
    if src.size < 16:
        return 1.0, 0.0

    scale, shift = _lstsq_1d(src, ref)

    if trim_ratio > 0.0 and src.size >= 64:
        residual = np.abs(src * scale + shift - ref)
        keep = residual <= np.quantile(residual, 1.0 - trim_ratio)
        if keep.sum() >= 16:
            scale, shift = _lstsq_1d(src[keep], ref[keep])

    if not np.isfinite(scale) or not np.isfinite(shift):
        return 1.0, 0.0
    return float(scale), float(shift)


def _lstsq_1d(src: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    # Closed-form solution of min ||a*src + b - ref||^2.
    src_mean = src.mean()
    ref_mean = ref.mean()
    var = ((src - src_mean) ** 2).mean()
    if var < 1e-12:
        return 1.0, 0.0
    cov = ((src - src_mean) * (ref - ref_mean)).mean()
    scale = cov / var
    shift = ref_mean - scale * src_mean
    return float(scale), float(shift)


def apply_scale_shift(src: np.ndarray, scale: float, shift: float) -> np.ndarray:
    return src.astype(np.float32) * np.float32(scale) + np.float32(shift)
