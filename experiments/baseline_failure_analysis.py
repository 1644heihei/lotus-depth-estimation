import argparse
import json
import os
import subprocess


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline depth inference and class-wise failure analysis.")
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="regression", choices=["regression", "generation"])
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--class_map_json", type=str, default=None)
    parser.add_argument("--boundary_width", type=int, default=8)
    parser.add_argument("--topk_worst", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    infer_out = os.path.join(args.output_dir, "baseline")
    run_cmd = [
        args.python_bin,
        "infer.py",
        "--pretrained_model_name_or_path", args.model,
        "--input_dir", args.input_dir,
        "--output_dir", infer_out,
        "--mode", args.mode,
        "--task_name", "depth",
        "--timestep", str(args.timestep),
        "--seed", str(args.seed),
    ]
    if args.half_precision:
        run_cmd.append("--half_precision")
    run(run_cmd)

    metrics_dir = os.path.join(args.output_dir, "analysis")
    os.makedirs(metrics_dir, exist_ok=True)
    eval_cmd = [
        args.python_bin,
        "evaluation/roi_depth_metrics.py",
        "--pred_dir", os.path.join(infer_out, "depth"),
        "--gt_dir", args.gt_dir,
        "--mask_dir", args.mask_dir,
        "--boundary_width", str(args.boundary_width),
        "--output_csv", os.path.join(metrics_dir, "baseline_per_sample.csv"),
        "--output_json", os.path.join(metrics_dir, "baseline_summary.json"),
    ]
    if args.class_map_json:
        eval_cmd += ["--class_map_json", args.class_map_json]
    run(eval_cmd)

    with open(os.path.join(metrics_dir, "baseline_summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    print("num_samples:", summary.get("num_samples", 0))
    print("overall:", summary.get("overall", {}))
    print("per_class keys:", list(summary.get("per_class", {}).keys()))


if __name__ == "__main__":
    main()
