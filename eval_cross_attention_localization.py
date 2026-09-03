#!/usr/bin/env python
"""Does Lotus still route a class-name token to that object's region?

Feeding class names at inference changed the output no more usefully than a fixed generic
string did, and I read that as text carrying no meaning for depth. That is evidence about
the untrained model only. Whether TRAINING the text channel could work turns on a
different question: in Stable Diffusion, cross-attention is the machinery that binds a
word to a region, and it is the reason a prompt controls image layout. Lotus inherited
that machinery and then fine-tuned with prompt="" throughout.

If the binding survived, training would be re-activating something that already exists,
and class names could localise after all - which would undo the "a name carries no
position" objection. If it was flattened by the empty-prompt fine-tuning, training starts
from nothing and has to pay the fine-tune tax for it.

One forward pass answers it. Cross-attention probabilities are captured for the token(s)
spelling each detected class, averaged over heads and layers, and scored by how much
attention mass lands inside that class's boxes:

    lift = mean attention inside the region / mean attention over the image

  lift ~ 1        no binding; the token is ignored spatially
  lift > 1        the token still points at its object

Two controls, since a lift above 1 has innocent explanations. OTHER gives the same token
another class's boxes in the same image - it catches a token that merely favours
foreground-shaped blobs. SHIFTED relocates the region, matching area and shape, as
everywhere else in this investigation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_object_oracle_ceiling import bbox_instance_masks
from eval_regressor_predepth_nyuv2 import list_nyu_pairs
from pipeline import LotusDPipeline
from utils.object_detection_cache import load_detections


def parse_args():
    p = argparse.ArgumentParser(description="Cross-attention localisation of class tokens.")
    p.add_argument(
        "--rgb_dir",
        type=str,
        default="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test",
    )
    p.add_argument("--detail_artifacts_dir", type=str, default="D:/lotus/data/nyuv2_detail_artifacts/test")
    p.add_argument("--core_model", type=str, default="jingheya/lotus-depth-d-v2-0-disparity")
    p.add_argument("--output_dir", type=str, default="output/eval_cross_attention")
    p.add_argument("--processing_res", type=int, default=768)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--detection_score_thr", type=float, default=0.5)
    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class CaptureProcessor:
    """Cross-attention processor that materialises the probabilities SDPA would hide."""

    def __init__(self, store, name):
        self.store, self.name = store, name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        inp_ndim = hidden_states.ndim
        if inp_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attn.norm_cross and encoder_hidden_states is not None:
            ctx = attn.norm_encoder_hidden_states(ctx)
        key, value = attn.to_k(ctx), attn.to_v(ctx)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        probs = attn.get_attention_scores(query, key, attention_mask)
        if encoder_hidden_states is not None:
            # (batch*heads, HW, 77) -> mean over heads
            heads = attn.heads
            self.store[self.name] = probs.detach().float().reshape(
                -1, heads, probs.shape[1], probs.shape[2]).mean(1)[0].cpu()

        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if inp_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def token_spans(tokenizer, prompt, names):
    """Token positions (in the padded sequence) spelling each class name."""
    ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids[0].tolist()
    spans = {}
    for n in names:
        sub = tokenizer(n, add_special_tokens=False).input_ids
        if not sub:
            continue
        for i in range(len(ids) - len(sub) + 1):
            if ids[i:i + len(sub)] == sub:
                spans[n] = list(range(i, i + len(sub)))
                break
    return spans


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = Path(args.rgb_dir)
    detail_root = Path(args.detail_artifacts_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = LotusDPipeline.from_pretrained(args.core_model, torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)

    store = {}
    procs = {}
    for name in pipe.unet.attn_processors:
        procs[name] = CaptureProcessor(store, name) if name.endswith("attn2.processor") \
            else pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(procs)

    pairs = list_nyu_pairs(rgb_dir)[: args.max_images]
    lifts = {"own": [], "other": [], "shifted": []}
    n_tok = 0

    for rgb_path, _ in tqdm(pairs, desc="cross_attn"):
        dets = [d for d in load_detections(rgb_path, detail_root) if d.score >= args.detection_score_thr]
        classes = sorted({d.class_name for d in dets})
        if len(classes) < 1:
            continue
        rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
        h, w = rgb_np.shape[:2]
        prompt = ", ".join(classes)
        spans = token_spans(pipe.tokenizer, prompt, classes)
        if not spans:
            continue

        image = torch.from_numpy(rgb_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        image = (image / 127.5 - 1.0).to(device)
        task_emb = torch.tensor([1, 0], device=device).float().unsqueeze(0)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
        store.clear()
        with torch.no_grad(), (nullcontext() if device.type == "mps"
                               else torch.autocast(device_type=device.type)):
            pipe(rgb_in=image, prompt=prompt, num_inference_steps=1,
                 generator=torch.Generator(device=device).manual_seed(args.seed),
                 output_type="np", timesteps=[args.timestep], task_emb=task_emb,
                 processing_res=args.processing_res, match_input_res=True)
        if not store:
            continue

        # average every cross-attention layer onto a common grid
        ar = w / h
        maps = []
        for m in store.values():
            hw = m.shape[0]
            lh = int(round((hw / ar) ** 0.5))
            lw = int(round(hw / max(lh, 1)))
            if lh * lw != hw:
                continue
            maps.append(F.interpolate(m.T.reshape(1, -1, lh, lw), size=(64, 64),
                                      mode="bilinear", align_corners=False))
        if not maps:
            continue
        att = torch.stack(maps).mean(0)[0]  # (77, 64, 64)

        per_class = {}
        for c in classes:
            m = bbox_instance_masks([d for d in dets if d.class_name == c], h, w)
            if not m:
                continue
            per_class[c] = np.any(np.stack(m), axis=0)

        for c, span in spans.items():
            if c not in per_class:
                continue
            a = att[span].mean(0).numpy()
            a = a / max(a.mean(), 1e-12)  # image-mean-normalised: lift reads directly
            small = lambda msk: np.array(Image.fromarray(msk.astype(np.uint8) * 255)
                                         .resize((64, 64), Image.NEAREST)) > 127
            own = small(per_class[c])
            if own.sum() < 8 or own.all():
                continue
            n_tok += 1
            lifts["own"].append(float(a[own].mean()))
            others = [per_class[o] for o in per_class if o != c]
            if others:
                om = small(np.any(np.stack(others), axis=0)) & ~own
                if om.sum() >= 8:
                    lifts["other"].append(float(a[om].mean()))
            dy, dx = int(rng.integers(13, 51)), int(rng.integers(13, 51))
            lifts["shifted"].append(float(a[np.roll(own, (dy, dx), axis=(0, 1))].mean()))

    summary = {"n_images": len(pairs), "n_tokens": n_tok,
               "lift": {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                            "n": len(v)} for k, v in lifts.items() if v}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nCross-attention localisation  {n_tok} class tokens over {len(pairs)} images")
    print("\nlift = attention inside the region / attention averaged over the image")
    print(f"\n{'region':<28}{'mean lift':>11}{'median':>9}{'n':>7}")
    print("-" * 55)
    for k, label in (("own", "the token's own boxes"),
                     ("other", "CONTROL other classes' boxes"),
                     ("shifted", "CONTROL same boxes, relocated")):
        if k in summary["lift"]:
            s = summary["lift"][k]
            print(f"{label:<28}{s['mean']:>11.3f}{s['median']:>9.3f}{s['n']:>7}")
    print("\n1.000 means the token is spatially ignored; above 1 means it still points.")
    print(f"\nSaved: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
