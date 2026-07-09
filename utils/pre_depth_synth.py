"""Synthetic pre-depth generation from GT depth for Phase-2 training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-6:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, device=x.device, dtype=x.dtype) - radius
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-8)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, ksize, ksize)
    kernel_2d = kernel_2d.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel_2d, padding=radius, groups=x.shape[1])


def synthesize_pre_depth(
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    dropout_p: float = 0.3,
    affine_scale_jitter: float = 0.10,
    affine_shift_jitter: float = 0.05,
    blur_sigma_min: float = 0.0,
    blur_sigma_max: float = 1.5,
    hole_keep_p: float = 0.92,
    noise_std: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create noisy pseudo pre-depth from GT depth.

    Args:
        gt_depth: [B,1,H,W] or [B,3,H,W], normalized target depth.
        valid_mask: optional [B,1,H,W] validity map.
    Returns:
        pre_depth: [B,1,H,W]
        valid_mask_out: [B,1,H,W] in {0,1}
    """
    if gt_depth.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(gt_depth.shape)}")

    if gt_depth.shape[1] == 3:
        pre = gt_depth.mean(dim=1, keepdim=True)
    elif gt_depth.shape[1] == 1:
        pre = gt_depth.clone()
    else:
        raise ValueError(f"Unsupported channel count: {gt_depth.shape[1]}")

    bsz = pre.shape[0]
    device = pre.device
    dtype = pre.dtype

    if valid_mask is None:
        vm = torch.ones((bsz, 1, pre.shape[-2], pre.shape[-1]), device=device, dtype=dtype)
    else:
        vm = valid_mask[:, :1].float().to(device=device, dtype=dtype)

    # Per-sample affine perturbation to mimic crop/global misalignment.
    scale = 1.0 + (torch.rand((bsz, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0) * affine_scale_jitter
    shift = (torch.rand((bsz, 1, 1, 1), device=device, dtype=dtype) * 2.0 - 1.0) * affine_shift_jitter
    pre = pre * scale + shift

    # Blur each sample with random sigma.
    sigmas = blur_sigma_min + (blur_sigma_max - blur_sigma_min) * torch.rand((bsz,), device=device, dtype=dtype)
    blurred = []
    for i in range(bsz):
        blurred.append(_gaussian_blur(pre[i : i + 1], float(sigmas[i].item())))
    pre = torch.cat(blurred, dim=0)

    # Random sparse holes.
    holes = (torch.rand_like(vm) < hole_keep_p).float()
    vm = vm * holes

    # Additive noise only on valid region.
    if noise_std > 0:
        pre = pre + torch.randn_like(pre) * noise_std * vm

    # Condition dropout: remove pre-depth entirely for some samples.
    if dropout_p > 0:
        drop = (torch.rand((bsz, 1, 1, 1), device=device) < dropout_p).float()
        keep = 1.0 - drop
        pre = pre * keep
        vm = vm * keep

    return pre, (vm > 0.5).float()

