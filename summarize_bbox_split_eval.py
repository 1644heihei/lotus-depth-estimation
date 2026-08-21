#!/usr/bin/env python
"""Aggregate per-checkpoint summary.json files from eval_bbox_split sweeps into one table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep_dir", type=str, required=True, help="Dir containing checkpoint-<step>/summary.json")
    p.add_argument("--csv_out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.sweep_dir)
    rows = []
    for summary_path in sorted(root.glob("checkpoint-*/summary.json")):
        m = re.search(r"checkpoint-(\d+)", summary_path.parent.name)
        step = int(m.group(1)) if m else -1
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "step": step,
                "abs_rel": data.get("abs_rel"),
                "roi_abs_rel": data.get("roi_abs_rel"),
                "bg_abs_rel": data.get("bg_abs_rel"),
                "delta1": data.get("delta1"),
                "num_images": data.get("num_images"),
            }
        )
    rows.sort(key=lambda r: r["step"])

    header = f"{'step':>6} | {'abs_rel':>9} | {'roi_abs_rel':>11} | {'bg_abs_rel':>10} | {'delta1':>7} | gap(roi-bg)"
    print(header)
    print("-" * len(header))
    for r in rows:
        roi = r["roi_abs_rel"]
        bg = r["bg_abs_rel"]
        gap = roi - bg if roi is not None and bg is not None else None
        roi_s = f"{roi:.6f}" if roi is not None else ""
        bg_s = f"{bg:.6f}" if bg is not None else ""
        gap_s = f"{gap:+.6f}" if gap is not None else ""
        print(
            f"{r['step']:>6} | {r['abs_rel']:.6f} | {roi_s:>11} | {bg_s:>10} | {r['delta1']:.4f} | {gap_s}"
        )

    if args.csv_out:
        import csv

        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "abs_rel", "roi_abs_rel", "bg_abs_rel", "delta1", "num_images"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
