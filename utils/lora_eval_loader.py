"""Attach a trained LoRA adapter to a pipeline for evaluation.

The text-conditioning experiment compares three training runs against each other and
against the untrained model (docs/text_conditioning_training_plan.md, section 5), so every
eval script has to be able to point at a checkpoint. eval_text_prompt_conditioning.py and
eval_cross_attention_localization.py were both written against the official model and load
no adapter at all, which would make the training unmeasurable.

Follows eval_regressor_predepth_nyuv2.py's existing call, factored out here so the
argument shape and the cache-invalidation rule stay identical across scripts.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def add_lora_args(parser) -> None:
    """--lora_path plus the run tag used to keep prediction caches apart."""
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help=(
            "Directory holding the trained adapter. Either a unet_lora/ directory or a "
            "checkpoint-N/ containing one. Omit to evaluate the official model."
        ),
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help=(
            "Name for this model under the prediction cache. Defaults to a slug of "
            "--lora_path, or 'base' with no adapter. Predictions from different weights "
            "MUST NOT share a cache directory."
        ),
    )


def resolve_lora_dir(lora_path: str | Path) -> Path:
    """Accept either unet_lora/ itself or the checkpoint directory holding it."""
    p = Path(lora_path)
    if (p / WEIGHT_NAME).is_file():
        return p
    nested = p / "unet_lora"
    if (nested / WEIGHT_NAME).is_file():
        return nested
    raise FileNotFoundError(f"No {WEIGHT_NAME} under {p} or {nested}")


def apply_lora(pipe, lora_path: str | Path | None) -> str | None:
    """Load the adapter into pipe.unet. Returns the directory used, or None."""
    if not lora_path:
        return None
    lora_dir = resolve_lora_dir(lora_path)
    pipe.unet.load_lora_adapter(str(lora_dir), weight_name=WEIGHT_NAME, prefix=None)
    logging.info("Loaded LoRA adapter from %s", lora_dir)
    return str(lora_dir)


def run_tag_for(args) -> str:
    """Cache subdirectory name for whichever weights are loaded.

    Predictions are cached by image path, so two models sharing a directory would silently
    read each other's outputs and the runs would look identical. Deriving the tag from the
    adapter path makes that impossible to do by accident.
    """
    if getattr(args, "run_tag", None):
        return args.run_tag
    if not getattr(args, "lora_path", None):
        return "base"
    p = Path(args.lora_path)
    parts = [x for x in (p.parent.name, p.name) if x and x != "unet_lora"]
    return re.sub(r"[^0-9A-Za-z._-]+", "_", "_".join(parts)) or "lora"
