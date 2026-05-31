import argparse
import csv
import glob
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis-ready markdown summary from experiment outputs.")
    parser.add_argument("--baseline_summary_json", type=str, required=True)
    parser.add_argument("--ablation_root", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--topk_examples", type=int, default=8)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def pick_best_ablation(ablation_root: str):
    summaries = glob.glob(os.path.join(ablation_root, "roi_fusion_*", "metrics", "summary.json"))
    candidates = []
    for path in summaries:
        data = load_json(path)
        score = try_float(data.get("roi_absrel_improvement_mean", 0.0))
        candidates.append((score, path, data))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0]


def load_top_examples(per_sample_csv: str, topk: int):
    if not os.path.exists(per_sample_csv):
        return []
    rows = []
    with open(per_sample_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rows.sort(key=lambda r: try_float(r.get("roi_absrel_improvement", "-inf"), -1e9), reverse=True)
    return rows[:topk]


def main():
    args = parse_args()
    baseline = load_json(args.baseline_summary_json)
    best = pick_best_ablation(args.ablation_root)

    lines = []
    lines.append("# Semantic-aware Depth Estimation Report")
    lines.append("")
    lines.append("## 1) Baseline summary")
    lines.append(f"- Samples: {baseline.get('num_samples', 0)}")
    overall = baseline.get("overall", {})
    if overall:
        lines.append(f"- ROI AbsRel: {overall.get('roi_absrel', 'n/a')}")
        lines.append(f"- ROI RMSE: {overall.get('roi_rmse', 'n/a')}")
        lines.append(f"- ROI Delta1: {overall.get('roi_delta1', 'n/a')}")
    lines.append("")
    lines.append("## 2) Class-wise failure tendencies")
    per_class = baseline.get("per_class", {})
    for class_name, metrics in per_class.items():
        lines.append(
            f"- {class_name}: roi_absrel={metrics.get('roi_absrel', 'n/a')}, boundary_absrel={metrics.get('boundary_absrel', 'n/a')}"
        )
    lines.append("")

    lines.append("## 3) ROI-fusion ablation result")
    if best is None:
        lines.append("- No ablation summary found under the specified root.")
    else:
        score, summary_path, summary_data = best
        run_dir = str(Path(summary_path).parents[1])
        lines.append(f"- Best run: `{run_dir}`")
        lines.append(f"- Mean ROI AbsRel improvement vs baseline: {score}")
        overall_best = summary_data.get("overall", {})
        lines.append(f"- ROI AbsRel (best): {overall_best.get('roi_absrel', 'n/a')}")
        lines.append(f"- ROI Delta1 (best): {overall_best.get('roi_delta1', 'n/a')}")

        top_rows = load_top_examples(os.path.join(run_dir, "metrics", "per_sample.csv"), args.topk_examples)
        lines.append("")
        lines.append("## 4) Recommended qualitative examples")
        if not top_rows:
            lines.append("- No per-sample CSV found for qualitative ranking.")
        else:
            for r in top_rows:
                img = r.get("image", "unknown")
                imp = r.get("roi_absrel_improvement", "n/a")
                cls = r.get("class_name", "unknown")
                lines.append(f"- `{img}` (class={cls}, roi_absrel_improvement={imp})")
            lines.append("")
            lines.append("Use the following files for side-by-side figures:")
            lines.append(f"- Baseline vis: `{Path(args.ablation_root) / 'baseline' / 'depth_vis'}`")
            lines.append(f"- Fused vis: `{Path(run_dir) / 'depth_vis'}`")
            lines.append(f"- ROI mask: `{Path(run_dir) / 'roi_mask'}`")

    lines.append("")
    lines.append("## 5) Discussion prompts")
    lines.append("- Which classes benefit most from ROI fusion?")
    lines.append("- Are gains concentrated near object boundaries?")
    lines.append("- How does semantic-mask-conditioned fine-tuning compare to inference-only fusion?")

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved report: {args.output_md}")


if __name__ == "__main__":
    main()
