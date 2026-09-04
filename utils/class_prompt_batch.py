"""Per-sample class-name prompts and spatial-bias inputs for the training loop.

train_lotus_d.py's loop runs 1.34 s/step and reads detections for every image in every
batch, so the two costs worth avoiding are re-parsing detection JSON for frames already
seen and re-tokenising the same handful of prompt strings. Hypersim's detector emits 27
classes, so a few thousand distinct prompts cover the entire dataset and both caches stay
small.

Kept out of the training script because the prompt has to match evaluation byte for byte;
utils/object_prompt.py is the single definition and this is the batched wrapper around it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.object_detection_cache import load_detections
from utils.object_prompt import (
    DEFAULT_MAX_CLASS_TOKENS,
    build_class_prompt,
    class_token_bias_inputs,
)


class ClassPromptCache:
    """Prompts and bias inputs per image path, memoised across steps."""

    def __init__(
        self,
        detections_root: str | Path,
        tokenizer,
        score_thr: float = 0.5,
        max_class_tokens: int = DEFAULT_MAX_CLASS_TOKENS,
    ):
        self.root = str(detections_root)
        self.tokenizer = tokenizer
        self.score_thr = score_thr
        self.max_class_tokens = max_class_tokens
        self._prompt: dict[str, str] = {}
        self._bias: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def check_root(self, sample_paths, min_hit_frac: float = 0.2) -> None:
        """Refuse to start when the detection root does not resolve.

        load_detections returns [] for a missing file, so a mis-rooted directory trains
        3000 steps on empty prompts and reports that text conditioning does nothing. That
        is the failure this experiment is least able to detect from its own results, so it
        is checked up front instead. The smoke run hit exactly this: the JSONs live under
        <root>/train/train/, and every prompt came out empty.

        The threshold is low because empty prompts are legitimate - 39.5% of Hypersim
        frames have no detection at all - so this catches a broken path, not a quiet one.
        """
        paths = [str(p) for p in sample_paths]
        if not paths:
            return
        hits = sum(1 for p in paths if self._detections(p))
        if hits / len(paths) < min_hit_frac:
            raise RuntimeError(
                f"Detections root {self.root!r} produced {hits}/{len(paths)} non-empty "
                f"results. Expected roughly 60% (Hypersim has 39.5% empty frames). "
                f"Check the path: probed e.g. {paths[0]!r}."
            )

    def _detections(self, path):
        return [d for d in load_detections(path, self.root) if d.score >= self.score_thr]

    def prompt_for(self, path) -> str:
        key = str(path)
        if key not in self._prompt:
            self._prompt[key] = build_class_prompt(self._detections(key), self.score_thr)
        return self._prompt[key]

    def _bias_for(self, path):
        key = str(path)
        if key not in self._bias:
            with Image.open(key) as im:
                w, h = im.size
            bbox, index, valid, _ = class_token_bias_inputs(
                self._detections(key), self.tokenizer, h, w,
                self.score_thr, self.max_class_tokens,
            )
            self._bias[key] = (bbox, index, valid)
        return self._bias[key]

    def encode_batch(
        self,
        paths,
        batch_size,
        tokenizer,
        text_encoder,
        device,
        prompt_dropout_p: float = 0.0,
        spatial_bias: bool = False,
        bias_dropout_p: float = 0.0,
        rng: "random.Random | None" = None,
    ):
        """Everything the training loop needs for one batch's text conditioning.

        Returns (encoder_hidden_states, encoder_attention_mask, bias_inputs, prompts).
        bias_inputs is None whenever the bias is off, dropped for this step, or no sample
        in the batch has detections.

        Gathered here rather than inlined because the pieces are interdependent - prompt
        dropout decides which samples may carry a bias, and the attention mask has to
        describe the same padded batch the encoder just produced - and the training loop
        is long enough that spreading them across it invites one to drift from the others.
        """
        import random as _random

        rnd = rng or _random
        prompts = [
            "" if rnd.random() < prompt_dropout_p else self.prompt_for(path)
            for path in paths
        ]
        # the RGB-reconstruction half is not the depth task and gets no conditioning
        prompts = prompts + [""] * (batch_size - len(prompts))

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]
        # Padding to a fixed width is what makes a batch of different prompts possible, but
        # LotusDPipeline encodes with padding="do_not_pad", so at inference the sequence is
        # only as long as the prompt ("sink" is 3 tokens, empty is 2). Without this mask the
        # model would learn to attend to 74 padding embeddings that are simply absent at
        # inference - CLIP's pad embedding is not zero - and the comparison would be against
        # a condition the model never sees in evaluation.
        mask = text_inputs.attention_mask.to(device)

        bias_inputs = None
        if spatial_bias and rnd.random() >= bias_dropout_p:
            # Dropped for whole batches rather than per sample: the bias enters through
            # cross_attention_kwargs, which the UNet applies uniformly.
            bias_inputs = self.bias_inputs_for(paths, batch_size, prompts, device)
        return hidden, mask, bias_inputs, prompts

    def bias_inputs_for(self, paths, batch_size, prompts, device):
        """Stack (bbox, token_index, token_mask) for the batch, or None if nothing valid.

        prompts is passed in rather than recomputed so that prompt dropout is honoured: a
        sample whose prompt was blanked this step has no class tokens to bias, and biasing
        toward an object the model was not told about would make the two conditions
        disagree about what the sample is.

        batch_size covers the RGB-reconstruction half, which is padded with invalid rows -
        it is not the depth task and gets no conditioning.
        """
        bbox = np.zeros((batch_size, self.max_class_tokens, 4), dtype=np.float32)
        index = np.zeros((batch_size, self.max_class_tokens), dtype=np.int64)
        valid = np.zeros((batch_size, self.max_class_tokens), dtype=bool)
        for i, path in enumerate(paths):
            if i >= batch_size or not prompts[i]:
                continue
            b, ix, v = self._bias_for(path)
            bbox[i], index[i], valid[i] = b, ix, v
        if not valid.any():
            return None
        return (
            torch.from_numpy(bbox).to(device),
            torch.from_numpy(index).to(device),
            torch.from_numpy(valid).to(device),
        )


def log_conditioning_once(seen: dict, logger, prompts, hidden, mask, bias_inputs) -> None:
    """Print the conditioning actually in flight, once for prompts and once for the bias.

    Two experiments in this repo were reported as started when the intended configuration
    had never taken effect, and the smoke run for this one trained on entirely empty
    prompts because the detection root resolved to nothing. None of those are visible in
    the loss, so the values themselves get printed rather than the flags that produced
    them. The bias needs its own flag because it is dropped half the time and so first
    appears on a later step than the prompts do.
    """
    if not seen.get("prompt"):
        seen["prompt"] = True
        for i, s in enumerate(prompts[:4]):
            logger.info("[prompt sample] %d: %r", i, s)
        logger.info(
            "[prompt sample] encoder_hidden_states=%s  attention_mask sums=%s",
            tuple(hidden.shape), mask.sum(dim=1)[:4].tolist(),
        )
    if bias_inputs is not None and not seen.get("bias"):
        seen["bias"] = True
        bbox, index, valid = bias_inputs
        counts = valid.sum(dim=1).tolist()
        # a sample that actually carries rows, so the indices can be checked against the
        # prompt string they were derived from
        j = next((i for i, c in enumerate(counts) if c), 0)
        logger.info("[bias sample] bbox=%s  valid rows/sample=%s", tuple(bbox.shape), counts)
        logger.info(
            "[bias sample] sample %d prompt=%r  token indices=%s  boxes=%s",
            j, prompts[j], index[j][valid[j]].tolist(),
            [[round(v, 3) for v in b] for b in bbox[j][valid[j]].tolist()],
        )
