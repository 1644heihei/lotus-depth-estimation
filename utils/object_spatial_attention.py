"""Spatial gating for object cross-attention tokens in Lotus-D UNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


def install_object_spatial_attention_processors(unet) -> None:
    """Replace attn2 processors with spatial-bias-aware cross-attention."""
    for name, module in unet.named_modules():
        if name.endswith("attn2") and hasattr(module, "set_processor"):
            module.set_processor(ObjectSpatialAttnProcessor())


def _attention_grid_size(
    hidden_states: torch.Tensor,
    input_ndim: int,
    height: int | None,
    width: int | None,
    ref_height: int | None,
    ref_width: int | None,
) -> tuple[int, int] | None:
    if input_ndim == 4 and height is not None and width is not None:
        return height, width

    seq_len = hidden_states.shape[1]
    if ref_height and ref_width and ref_height * ref_width > 0:
        scale = (ref_height * ref_width / seq_len) ** 0.5
        grid_h = max(1, int(round(ref_height / scale)))
        grid_w = max(1, int(round(ref_width / scale)))
        if grid_h * grid_w != seq_len:
            grid_w = max(1, seq_len // grid_h)
        if grid_h * grid_w != seq_len:
            grid_h, grid_w = seq_len, 1
        return grid_h, grid_w

    side = int(seq_len**0.5)
    if side * side == seq_len:
        return side, side
    return seq_len, 1


def build_object_spatial_attention_bias(
    continuous_features: torch.Tensor,
    object_mask: torch.Tensor,
    latent_height: int,
    latent_width: int,
    num_text_tokens: int,
    *,
    inside_bias: float = 10.0,
    outside_bias: float = -2.0,
) -> torch.Tensor | None:
    """Build additive cross-attention bias for object tokens only.

    Args:
        continuous_features: [B, K, D] with normalized (cx, cy, w, h) in first 4 dims.
        object_mask: [B, K] bool/float mask for real objects.
        latent_height/latent_width: UNet latent grid size.
        num_text_tokens: number of CLIP text tokens prepended before object tokens.

    Returns:
        Tensor [B, 1, Q, num_text_tokens + K] or None when no valid objects exist.
    """
    if object_mask is None or not bool(object_mask.any()):
        return None

    batch_size, num_objects, feat_dim = continuous_features.shape
    if feat_dim < 4:
        raise ValueError(
            f"continuous_features need at least 4 dims for bbox, got {feat_dim}"
        )

    device = continuous_features.device
    dtype = continuous_features.dtype
    mask = object_mask.to(device=device, dtype=dtype)
    bbox = continuous_features[..., :4].clamp(0.0, 1.0)

    grid_y = (torch.arange(latent_height, device=device, dtype=dtype) + 0.5) / float(
        latent_height
    )
    grid_x = (torch.arange(latent_width, device=device, dtype=dtype) + 0.5) / float(
        latent_width
    )
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    yy = yy.reshape(1, 1, -1)
    xx = xx.reshape(1, 1, -1)

    cx = bbox[..., 0:1]
    cy = bbox[..., 1:2]
    bw = bbox[..., 2:3].clamp(min=1e-4)
    bh = bbox[..., 3:4].clamp(min=1e-4)

    dx = (xx - cx) / (0.5 * bw + 1e-4)
    dy = (yy - cy) / (0.5 * bh + 1e-4)
    radial = dx.square() + dy.square()
    inside_score = torch.exp(-0.5 * radial)
    object_bias = inside_bias * inside_score + outside_bias * (1.0 - inside_score)
    object_bias = object_bias * mask.unsqueeze(-1)

    query_len = latent_height * latent_width
    key_len = num_text_tokens + num_objects
    bias = object_bias.new_zeros(batch_size, 1, query_len, key_len)
    bias[:, :, :, num_text_tokens:] = object_bias.transpose(1, 2).unsqueeze(1)
    return bias


def object_cross_attention_kwargs(
    continuous_features: torch.Tensor | None,
    object_mask: torch.Tensor | None,
    text_embeddings: torch.Tensor,
    latent_height: int | None = None,
    latent_width: int | None = None,
    *,
    enabled: bool = True,
) -> dict:
    if not enabled or continuous_features is None or object_mask is None:
        return {}
    if not bool(object_mask.any()):
        return {}
    kwargs = {
        "object_spatial_features": continuous_features,
        "object_spatial_mask": object_mask,
        "object_spatial_num_text_tokens": int(text_embeddings.shape[1]),
    }
    if latent_height is not None and latent_width is not None:
        kwargs["object_spatial_ref_height"] = latent_height
        kwargs["object_spatial_ref_width"] = latent_width
    return kwargs


class ObjectSpatialAttnProcessor(AttnProcessor2_0):
    """AttnProcessor2_0 with optional additive object spatial bias."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        object_spatial_bias: torch.Tensor | None = None,
        object_spatial_features: torch.Tensor | None = None,
        object_spatial_mask: torch.Tensor | None = None,
        object_spatial_num_text_tokens: int | None = None,
        object_spatial_ref_height: int | None = None,
        object_spatial_ref_width: int | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        input_ndim = hidden_states.ndim
        channel = height = width = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
        else:
            batch_size = hidden_states.shape[0]

        spatial_bias = object_spatial_bias
        if spatial_bias is None and object_spatial_features is not None:
            grid_size = _attention_grid_size(
                hidden_states,
                input_ndim,
                height,
                width,
                object_spatial_ref_height,
                object_spatial_ref_width,
            )
            if grid_size is not None:
                grid_h, grid_w = grid_size
                spatial_bias = build_object_spatial_attention_bias(
                    object_spatial_features,
                    object_spatial_mask,
                    grid_h,
                    grid_w,
                    object_spatial_num_text_tokens or 0,
                )

        if encoder_hidden_states is None or spatial_bias is None:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(
                1, 2
            )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask,
                encoder_hidden_states.shape[1],
                batch_size,
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        spatial_bias = spatial_bias.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        if spatial_bias.shape[1] == 1:
            spatial_bias = spatial_bias.expand(-1, attn.heads, -1, -1)
        if attention_mask is None:
            attention_mask = spatial_bias
        else:
            attention_mask = attention_mask + spatial_bias

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor
