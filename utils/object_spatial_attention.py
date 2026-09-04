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


def build_class_token_spatial_bias(
    bbox: torch.Tensor,
    token_index: torch.Tensor,
    token_mask: torch.Tensor,
    latent_height: int,
    latent_width: int,
    num_text_tokens: int,
    *,
    inside_bias: float = 10.0,
    outside_bias: float = -2.0,
) -> torch.Tensor | None:
    """Bias the CLIP text tokens that spell a class name toward that object's region.

    build_object_spatial_attention_bias above writes into APPENDED object tokens and
    leaves every text column at zero. Option C in docs/text_conditioning_training_plan.md
    needs the opposite: the class name's own CLIP tokens get pointed at the object, so the
    token carries pretrained semantics AND a given location, and training only has to
    learn to use it. Attention binding measured at 1.9% of full strength is what makes
    handing the location over worthwhile rather than waiting for it to be learned.

    Args:
        bbox: [B, K, 4] normalised (cx, cy, w, h), one row per (detection, token) pair.
        token_index: [B, K] CLIP position each row writes to.
        token_mask: [B, K] which rows are real rather than padding.
        num_text_tokens: width of the text sequence, normally the tokenizer's 77.

    Returns:
        [B, 1, Q, num_text_tokens], or None when no row is valid. Columns for tokens no
        detection claims - BOS, EOS, separators, padding - stay exactly zero, so the bias
        never disturbs attention to the rest of the prompt.
    """
    if token_mask is None or not bool(token_mask.any()):
        return None

    batch_size, num_rows, feat_dim = bbox.shape
    if feat_dim < 4:
        raise ValueError(f"bbox needs at least 4 dims, got {feat_dim}")

    device, dtype = bbox.device, bbox.dtype
    box = bbox[..., :4].clamp(0.0, 1.0)
    query_len = latent_height * latent_width

    grid_y = (torch.arange(latent_height, device=device, dtype=dtype) + 0.5) / float(latent_height)
    grid_x = (torch.arange(latent_width, device=device, dtype=dtype) + 0.5) / float(latent_width)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    yy = yy.reshape(1, 1, -1)
    xx = xx.reshape(1, 1, -1)

    cx, cy = box[..., 0:1], box[..., 1:2]
    bw = box[..., 2:3].clamp(min=1e-4)
    bh = box[..., 3:4].clamp(min=1e-4)
    dx = (xx - cx) / (0.5 * bw + 1e-4)
    dy = (yy - cy) / (0.5 * bh + 1e-4)
    inside_score = torch.exp(-0.5 * (dx.square() + dy.square()))
    value = inside_bias * inside_score + outside_bias * (1.0 - inside_score)  # [B, K, Q]

    # Several instances of one class point at the same token, so rows collide. Take the
    # max rather than a sum or mean: the token should be free to attend wherever any one
    # of them is, not pulled to a blurred midpoint between two chairs.
    neg_inf = torch.finfo(dtype).min
    value = value.masked_fill(~token_mask.to(torch.bool).unsqueeze(-1), neg_inf)
    packed = value.new_full((batch_size, num_text_tokens, query_len), neg_inf)
    index = token_index.clamp(0, num_text_tokens - 1).long().unsqueeze(-1).expand(
        batch_size, num_rows, query_len
    )
    packed.scatter_reduce_(1, index, value, reduce="amax", include_self=True)

    packed = torch.where(packed <= neg_inf, torch.zeros_like(packed), packed)
    return packed.transpose(1, 2).unsqueeze(1)


def class_token_cross_attention_kwargs(
    bbox: torch.Tensor | None,
    token_index: torch.Tensor | None,
    token_mask: torch.Tensor | None,
    num_text_tokens: int | None = None,
    latent_height: int | None = None,
    latent_width: int | None = None,
    *,
    enabled: bool = True,
) -> dict:
    """Inputs for the per-layer bias, not the bias itself.

    Q differs per attention block (64^2 down to 8^2 when training at 512), so the caller
    cannot build one tensor and pass it down; the processor builds it for its own grid.

    The ref_height/ref_width hints keep the existing object-bias names because
    _attention_grid_size is shared: they say how to factor a flattened sequence back into
    a grid, which matters whenever the image is not square. NYUv2 at 768 gives a 72x96
    latent, so without them a 6912-long sequence would be read as 83x83 and the bias would
    land on the wrong pixels.
    """
    if not enabled or bbox is None or token_mask is None:
        return {}
    if not bool(token_mask.any()):
        return {}
    kwargs = {
        "class_token_bbox": bbox,
        "class_token_index": token_index,
        "class_token_mask": token_mask,
    }
    # Leave num_text_tokens unset to take the runtime sequence width. LotusDPipeline
    # encodes prompts with padding="do_not_pad", so "sink" is 3 tokens, not 77; a bias
    # built 77 wide would not broadcast against a 3-wide key and the forward pass dies.
    if num_text_tokens is not None:
        kwargs["class_token_num_text_tokens"] = int(num_text_tokens)
    if latent_height is not None and latent_width is not None:
        kwargs["object_spatial_ref_height"] = latent_height
        kwargs["object_spatial_ref_width"] = latent_width
    return kwargs


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
        class_token_bbox: torch.Tensor | None = None,
        class_token_index: torch.Tensor | None = None,
        class_token_mask: torch.Tensor | None = None,
        class_token_num_text_tokens: int | None = None,
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
        if spatial_bias is None and (
            object_spatial_features is not None or class_token_bbox is not None
        ):
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
                if object_spatial_features is not None:
                    spatial_bias = build_object_spatial_attention_bias(
                        object_spatial_features,
                        object_spatial_mask,
                        grid_h,
                        grid_w,
                        object_spatial_num_text_tokens or 0,
                    )
                if class_token_bbox is not None:
                    class_bias = build_class_token_spatial_bias(
                        class_token_bbox,
                        class_token_index,
                        class_token_mask,
                        grid_h,
                        grid_w,
                        class_token_num_text_tokens or encoder_hidden_states.shape[1],
                    )
                    if class_bias is not None:
                        # The two write to disjoint columns - object tokens are appended
                        # after the text - so adding is safe when both are in play.
                        if spatial_bias is None:
                            spatial_bias = class_bias
                        else:
                            spatial_bias[..., : class_bias.shape[-1]] += class_bias

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
