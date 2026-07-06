import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.evaluation import evaluation_depth
from pipeline import LotusDPipeline


def build_gen_depth(pipe, timestep=999):
    def gen_depth(rgb_in, _pipe_unused=None, prompt="", num_inference_steps=1):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(pipe.device.type)

        with autocast_ctx:
            rgb_input = rgb_in / 255.0 * 2.0 - 1.0
            task_emb = torch.tensor([1, 0], device=pipe.device).float().unsqueeze(0)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
            pred_depth = pipe(
                rgb_in=rgb_input,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                output_type="np",
                timesteps=[timestep],
                task_emb=task_emb,
                processing_res=0,
            ).images[0]
        return pred_depth.mean(axis=-1)

    return gen_depth


def run_one(model_path, out_dir, base_test_data_dir, half=True, disparity=True):
    dtype = torch.float16 if half else torch.float32
    pipe = LotusDPipeline.from_pretrained(model_path, torch_dtype=dtype).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    config = os.path.join(base_test_data_dir, "depth", "configs", "data_nyu_test.yaml")
    alignment = "least_square_disparity" if disparity else "least_square"
    tracker = evaluation_depth(
        output_dir=out_dir,
        dataset_config=config,
        base_data_dir=os.path.join(base_test_data_dir, "depth"),
        eval_mode="generate_prediction",
        gen_prediction=build_gen_depth(pipe),
        pipeline=pipe,
        alignment=alignment,
        processing_res=None,
    )
    return tracker.result()


def main():
    base_test_data_dir = os.path.join("datasets", "eval")
    out_root = os.path.join("output", "eval_compare_nyuv2")
    os.makedirs(out_root, exist_ok=True)

    official = run_one(
        model_path="jingheya/lotus-depth-d-v2-0-disparity",
        out_dir=os.path.join(out_root, "official"),
        base_test_data_dir=base_test_data_dir,
        half=True,
        disparity=True,
    )
    ours = run_one(
        model_path=os.path.join("output", "train-lotus-d-depth-semantic-mask-bsz8"),
        out_dir=os.path.join(out_root, "ours"),
        base_test_data_dir=base_test_data_dir,
        half=True,
        disparity=True,
    )

    report = os.path.join(out_root, "summary.txt")
    keys = [
        "abs_relative_difference",
        "rmse_linear",
        "silog_rmse",
        "delta1_acc",
        "delta2_acc",
        "delta3_acc",
    ]
    lines = ["NYUv2 comparison (official vs ours)\n", ""]
    lines.append("metric,official,ours")
    for k in keys:
        lines.append(f"{k},{official.get(k)},{ours.get(k)}")
    text = "\n".join(lines) + "\n"
    with open(report, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
