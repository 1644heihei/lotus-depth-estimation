"""Utilities for Phase-2 pre-depth latent fusion.

This module provides:
- UNet input expansion with zero initialization
- pre-depth map encoding into VAE latent space
- latent-space valid-mask preparation
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models import UNet2DConditionModel


def expand_unet_conv_in(
    unet: UNet2DConditionModel,
    extra_in_channels: int,
    zero_init: bool = True,
) -> None:
    """Expand `unet.conv_in` input channels while keeping legacy behavior.

    Args:
        unet: Diffusers UNet2DConditionModel.
        extra_in_channels: Number of channels to append to the input.
        zero_init: If True, appended kernel slices are zero-initialized.
            This guarantees that feeding zero additional channels yields
            identical outputs to the original model at initialization.
    """
    if extra_in_channels <= 0:
        return

    conv = unet.conv_in
    if conv is None:
        raise ValueError("UNet has no conv_in layer.")

    old_in = conv.in_channels
    new_in = old_in + extra_in_channels
    if old_in == new_in:
        return

    new_conv = torch.nn.Conv2d(
        in_channels=new_in,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    with torch.no_grad():
        new_conv.weight[:, :old_in].copy_(conv.weight)
        if zero_init:
            new_conv.weight[:, old_in:].zero_()
        else:
            torch.nn.init.kaiming_normal_(new_conv.weight[:, old_in:], mode="fan_in", nonlinearity="leaky_relu")
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    unet.conv_in = new_conv
    unet.config.in_channels = new_in
    # Persist config update for save_pretrained/load_pretrained cycle.
    if hasattr(unet, "register_to_config"):
        unet.register_to_config(in_channels=new_in)


def ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    """Convert [B,1,H,W] tensor to [B,3,H,W] by replication."""
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected channel=1 or 3, got {x.shape[1]}")


def encode_pre_depth_latents(
    vae,
    pre_depth: torch.Tensor,
    *,
    scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    """Encode pre-depth map to latent with same VAE path as RGB/depth.

    `pre_depth` is expected to be in the same normalized range used in training
    (typically [-1, 1] for Lotus-D inputs/targets).
    """
    pre_depth_3ch = ensure_3ch(pre_depth)
    lat = vae.encode(pre_depth_3ch).latent_dist.sample()
    sf = scaling_factor if scaling_factor is not None else vae.config.scaling_factor
    return lat * sf


def downsample_valid_mask(
    valid_mask: torch.Tensor,
    target_hw: Tuple[int, int],
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Downsample valid mask to latent resolution.

    Args:
        valid_mask: [B,1,H,W] or [B,C,H,W], values in [0,1] or bool.
        target_hw: (H_lat, W_lat)
    Returns:
        [B,1,H_lat,W_lat] float tensor in {0,1}.
    """
    if valid_mask.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(valid_mask.shape)}")
    if valid_mask.shape[1] > 1:
        valid_mask = valid_mask[:, :1]
    vm = valid_mask.float()
    vm = F.interpolate(vm, size=target_hw, mode="nearest")
    return (vm >= threshold).float()

