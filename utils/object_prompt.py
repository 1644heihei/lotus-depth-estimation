"""Class-name prompts for Lotus's unused CLIP text channel, shared by training and eval.

Lotus inherits Stable Diffusion's text encoder and feeds prompt embeddings to the UNet as
encoder_hidden_states, but every call site passes prompt="". docs/text_conditioning_
training_plan.md trains that channel to carry object semantics, on the measured deficit
that Lotus recovers 31.2% of true depth discontinuities lying on an image edge but only
10.5% of those in visually flat regions - where a human still knows the step is there
because they know it is a chair.

Everything about the prompt lives here rather than in the training script, because the
prompt string has to be byte-identical between training and evaluation. A model trained
on "chair, sink" and evaluated on "a chair, a sink" is being tested on a condition it
never saw, and the failure would look like the idea not working.

The spatial-bias inputs live here too (option C in the plan): the bias is applied to the
CLIP token positions spelling each class, so the token indices and the boxes have to be
derived from the same tokenisation as the prompt itself.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

DEFAULT_SCORE_THR = 0.5
DEFAULT_MAX_CLASS_TOKENS = 16


def class_names(detections: Iterable, score_thr: float = DEFAULT_SCORE_THR) -> list[str]:
    """Distinct class names above the threshold, in a stable order.

    Sorted rather than detection order so that the same scene always produces the same
    string - detection order depends on confidence ranking and would make the prompt
    unstable across YOLO versions.
    """
    return sorted({d.class_name for d in detections if d.score >= score_thr})


def build_class_prompt(detections: Iterable, score_thr: float = DEFAULT_SCORE_THR) -> str:
    """The prompt itself. Empty string when nothing was detected.

    An empty prompt is the right answer for an undetected scene, not a special case: it
    is exactly what Lotus was fine-tuned on, and 39.5% of Hypersim frames land here, which
    doubles as implicit prompt dropout.
    """
    return ", ".join(class_names(detections, score_thr))


def class_token_spans(tokenizer, prompt: str, names: Sequence[str]) -> dict[str, list[int]]:
    """Token positions in the padded sequence that spell each class name.

    Matched by subsequence rather than by re-tokenising pieces, because CLIP's BPE merges
    across word boundaries: "dining table" tokenises differently alone than it does inside
    a longer prompt. Searching the padded ids for the standalone tokenisation is exact
    for the comma-separated format built above.
    """
    ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()

    spans: dict[str, list[int]] = {}
    for name in names:
        sub = tokenizer(name, add_special_tokens=False).input_ids
        if not sub:
            continue
        for i in range(len(ids) - len(sub) + 1):
            if ids[i : i + len(sub)] == sub:
                spans[name] = list(range(i, i + len(sub)))
                break
    return spans


def class_token_bias_inputs(
    detections: Iterable,
    tokenizer,
    image_height: int,
    image_width: int,
    score_thr: float = DEFAULT_SCORE_THR,
    max_tokens: int = DEFAULT_MAX_CLASS_TOKENS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Per-entry boxes and CLIP token positions for the spatial bias (option C).

    Returns (bbox[max_tokens, 4], token_index[max_tokens], valid[max_tokens], prompt).
    bbox is normalised (cx, cy, w, h) to match build_object_spatial_attention_bias's
    existing convention.

    One entry per (detection, token) pair, so indices repeat in two ways and both are
    intended. A class name spanning several tokens ("dining table") contributes one entry
    per token with the same box, so the whole name gets biased. Several instances of one
    class point at the same token, and the consumer resolves that with a max over entries
    - "attend wherever any one of them is" rather than a blurred average of both.

    Boxes come from detections rather than segmentation masks: hypersim_sem_masks stores
    one merged mask per frame, not per instance, and at the latent resolutions the bias
    acts on (64x64 and below when training at 512) a box and a silhouette differ by a few
    cells. See the plan's section 0.
    """
    dets = [d for d in detections if d.score >= score_thr]
    names = class_names(dets, score_thr)
    prompt = ", ".join(names)

    bbox = np.zeros((max_tokens, 4), dtype=np.float32)
    index = np.zeros((max_tokens,), dtype=np.int64)
    valid = np.zeros((max_tokens,), dtype=bool)
    if not dets:
        return bbox, index, valid, prompt

    spans = class_token_spans(tokenizer, prompt, names)
    h = max(float(image_height), 1.0)
    w = max(float(image_width), 1.0)

    n = 0
    for det in dets:
        span = spans.get(det.class_name)
        if not span:
            continue
        x1, y1, x2, y2 = (float(v) for v in det.bbox)
        box = (
            np.clip((x1 + x2) / 2.0 / w, 0.0, 1.0),
            np.clip((y1 + y2) / 2.0 / h, 0.0, 1.0),
            np.clip(abs(x2 - x1) / w, 1e-4, 1.0),
            np.clip(abs(y2 - y1) / h, 1e-4, 1.0),
        )
        for tok in span:
            if n >= max_tokens:
                # Hypersim averages 1.60 detections per frame and peaks at 9, so the
                # default cap is not reached in practice; truncate rather than fail.
                return bbox, index, valid, prompt
            bbox[n] = box
            index[n] = tok
            valid[n] = True
            n += 1
    return bbox, index, valid, prompt
