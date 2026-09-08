"""Keep an expanded conv_in trainable, saved, and loaded alongside a LoRA adapter.

expand_unet_conv_in appends zero-initialised input slices so a run starts at exactly the
pretrained model. Under LoRA that guarantee turns into a trap: the backbone is frozen right
after the expansion and the adapter targets attention projections, so the new slices stay
zero and stay frozen. The extra channel is then multiplied by zero on every step of every
run - the conditioning is fed in, logged, and discarded.

That is what happened to the first contour-conditioning attempt. Three 80-minute runs
trained on what was effectively the same model, and the input logging did not catch it
because logging proves the tensor was passed, not that it reached the output.

So this module owns three things that have to agree:
  * unfreeze  the new slices need requires_grad, or the optimizer never sees them
  * save      save_lora_adapter writes attention weights only, so conv_in needs its own file
  * load      evaluation builds the pipeline from the base model, whose conv_in is the
              original width, and must expand and repopulate it before the adapter applies

and one check that does not rely on any of the above being right: run the model twice with
different conditioning and confirm the outputs differ.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

FILENAME = "conv_in.safetensors"


def unfreeze_conv_in(unet, expected_in_channels: int) -> int:
    """Make conv_in trainable after an expansion. Returns its parameter count."""
    conv = unet.conv_in
    if conv.in_channels != expected_in_channels:
        raise ValueError(
            f"conv_in has {conv.in_channels} input channels, expected "
            f"{expected_in_channels}. The expansion did not run, or ran twice."
        )
    n = 0
    for p in conv.parameters():
        p.requires_grad_(True)
        p.data = p.data.float()  # LoRA params are kept fp32; match them
        n += p.numel()
    return n


def save_conv_in(unet, directory) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    path = d / FILENAME
    save_file({k: v.detach().cpu().contiguous()
               for k, v in unet.conv_in.state_dict().items()}, str(path))
    return path


def load_conv_in(unet, directory, strict: bool = True) -> bool:
    """Restore conv_in from a directory holding FILENAME. False when absent."""
    path = Path(directory) / FILENAME
    if not path.is_file():
        if strict:
            raise FileNotFoundError(
                f"{path} not found. A contour-conditioned adapter needs its conv_in; "
                f"without it the extra channels stay at their initial value and the "
                f"conditioning is silently ignored."
            )
        return False
    state = load_file(str(path))
    w = state["weight"]
    if w.shape[1] != unet.conv_in.in_channels:
        from utils.pre_depth_fusion import expand_unet_conv_in

        expand_unet_conv_in(unet, w.shape[1] - unet.conv_in.in_channels, zero_init=True)
    unet.conv_in.load_state_dict(
        {k: v.to(dtype=unet.conv_in.weight.dtype) for k, v in state.items()}
    )
    logging.info("Loaded conv_in (%d input channels) from %s", w.shape[1], path)
    return True


def extra_channel_energy(unet, base_in_channels: int) -> float:
    """Largest absolute weight on the appended input channels.

    Zero means training never moved them and whatever was fed through those channels had
    no effect - the failure this module exists to prevent, checked directly rather than
    inferred from the code being correct.
    """
    w = unet.conv_in.weight
    if w.shape[1] <= base_in_channels:
        return 0.0
    return float(w[:, base_in_channels:].abs().max())
