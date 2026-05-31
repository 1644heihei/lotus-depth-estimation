import argparse
import os
import subprocess
from pathlib import Path
from typing import List


def parse_weights(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run(cmd: List[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline + ROI-fusion ablations.")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="regression", choices=["regression", "generation"])
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--weights", type=str, default="0.4,0.7,1.0")
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--class_map_json", type=str, default=None)
    parser.add_argument("--metrics_boundary_width", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    baseline_out = os.path.join(args.output_root, "baseline")
    baseline_cmd = [
        args.python_bin,
        "infer.py",
        "--pretrained_model_name_or_path", args.model,
        "--input_dir", args.input_dir,
        "--output_dir", baseline_out,
        "--mode", args.mode,
        "--task_name", "depth",
        "--timestep", str(args.timestep),
        "--seed", str(args.seed),
    ]
    if args.half_precision:
        baseline_cmd.append("--half_precision")
    run(baseline_cmd)

    weights = parse_weights(args.weights)
    for w in weights:
        tag = str(w).replace(".", "p")
        out_dir = os.path.join(args.output_root, f"roi_fusion_w{tag}")
        cmd = [
            args.python_bin,
            "infer_roi_fusion.py",
            "--pretrained_model_name_or_path", args.model,
            "--input_dir", args.input_dir,
            "--output_dir", out_dir,
            "--mode", args.mode,
            "--task_name", "depth",
            "--timestep", str(args.timestep),
            "--fusion_weight", str(w),
            "--seed", str(args.seed),
        ]
        if args.mask_dir:
            cmd += ["--roi_mask_dir", args.mask_dir]
        if args.half_precision:
            cmd.append("--half_precision")
        run(cmd)

        if args.gt_dir and args.mask_dir:
            eval_out_dir = os.path.join(out_dir, "metrics")
            os.makedirs(eval_out_dir, exist_ok=True)
            eval_cmd = [
                args.python_bin,
                "evaluation/roi_depth_metrics.py",
                "--pred_dir", os.path.join(out_dir, "depth"),
                "--gt_dir", args.gt_dir,
                "--mask_dir", args.mask_dir,
                "--baseline_pred_dir", os.path.join(baseline_out, "depth"),
                "--boundary_width", str(args.metrics_boundary_width),
                "--output_csv", os.path.join(eval_out_dir, "per_sample.csv"),
                "--output_json", os.path.join(eval_out_dir, "summary.json"),
            ]
            if args.class_map_json:
                eval_cmd += ["--class_map_json", args.class_map_json]
            run(eval_cmd)

    print(f"Completed ablation runs under: {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
