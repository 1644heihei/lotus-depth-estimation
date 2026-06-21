from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_mask_shape(
    semantic_mask: Optional[torch.FloatTensor],
    batch_size: int,
    target_hw: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.FloatTensor:
    h, w = target_hw
    if semantic_mask is None:
        return torch.zeros((batch_size, 1, h, w), device=device, dtype=dtype)

    if semantic_mask.ndim != 4:
        raise ValueError(f"semantic_mask must be 4D [B,C,H,W], got shape={semantic_mask.shape}")

    if semantic_mask.shape[0] != batch_size:
        raise ValueError(
            f"semantic_mask batch ({semantic_mask.shape[0]}) != rgb batch ({batch_size})"
        )

    if semantic_mask.shape[1] > 1:
        semantic_mask = semantic_mask[:, :1]

    semantic_mask = semantic_mask.to(device=device, dtype=dtype)
    semantic_mask = torch.clamp(semantic_mask, 0.0, 1.0)
    semantic_mask = F.interpolate(semantic_mask, size=(h, w), mode="bilinear", align_corners=False)
    return semantic_mask


def concat_rgb_and_mask(
    rgb_in: torch.FloatTensor,
    semantic_mask: Optional[torch.FloatTensor],
    mask_value_when_missing: float = 0.0,
) -> torch.FloatTensor:
    """Build 4-channel early-fusion tensor [RGB, mask]."""
    if rgb_in.ndim != 4:
        raise ValueError(f"rgb_in must be 4D [B,C,H,W], got shape={rgb_in.shape}")
    if rgb_in.shape[1] == 4:
        return rgb_in
    if rgb_in.shape[1] != 3:
        raise ValueError(f"rgb_in channel must be 3 or 4, got {rgb_in.shape[1]}")

    sem = _normalize_mask_shape(
        semantic_mask=semantic_mask,
        batch_size=rgb_in.shape[0],
        target_hw=rgb_in.shape[-2:],
        device=rgb_in.device,
        dtype=rgb_in.dtype,
    )
    # Match RGB range [-1, 1].
    sem = sem * 2.0 - 1.0
    if semantic_mask is None and mask_value_when_missing != 0.0:
        sem = torch.full_like(sem, fill_value=mask_value_when_missing)
    return torch.cat([rgb_in, sem], dim=1)


def append_constant_channel(x: torch.FloatTensor, value: float = 0.0) -> torch.FloatTensor:
    """Append one constant channel to 3-channel tensors (no-op for 4-channel)."""
    if x.ndim != 4:
        raise ValueError(f"Input must be 4D [B,C,H,W], got shape={x.shape}")
    if x.shape[1] == 4:
        return x
    if x.shape[1] != 3:
        raise ValueError(f"Expected channel=3 or 4, got {x.shape[1]}")
    const = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), value, device=x.device, dtype=x.dtype)
    return torch.cat([x, const], dim=1)


def enable_vae_early_fusion(vae) -> bool:
    """
    Expand VAE encoder input from 3 to 4 channels.
    Returns True if conversion happened, False if already compatible.
    """
    conv_in = vae.encoder.conv_in
    if conv_in.in_channels == 4:
        return False
    if conv_in.in_channels != 3:
        raise ValueError(f"Unsupported VAE encoder input channels: {conv_in.in_channels}")

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=conv_in.out_channels,
        kernel_size=conv_in.kernel_size,
        stride=conv_in.stride,
        padding=conv_in.padding,
        dilation=conv_in.dilation,
        groups=conv_in.groups,
        bias=conv_in.bias is not None,
        padding_mode=conv_in.padding_mode,
        device=conv_in.weight.device,
        dtype=conv_in.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = conv_in.weight
        new_conv.weight[:, 3:4] = conv_in.weight.mean(dim=1, keepdim=True)
        if conv_in.bias is not None:
            new_conv.bias.copy_(conv_in.bias)

    vae.encoder.conv_in = new_conv
    # Keep config consistent for save/load.
    if hasattr(vae, "register_to_config"):
        vae.register_to_config(in_channels=4)
    return True
