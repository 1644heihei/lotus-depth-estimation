#!/usr/bin/env python
"""Compare per-sample metrics across eval conditions."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_metrics(eval_root: Path, name: str) -> dict[str, dict]:
    rows = {}
    with (eval_root / name / "per_sample_metrics.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["filename"]] = {
                "abs_rel": float(row["abs_relative_difference"]),
                "delta1": float(row["delta1_acc"]),
                "n_det": int(row["n_detections"]),
            }
    return rows


def bucket(n: int) -> str:
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def main():
    eval_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "output/eval-object-attention-8k-6500"
    )
    official = load_metrics(eval_root, "official")
    att_off = load_metrics(eval_root, "attention-off")
    att_on = load_metrics(eval_root, "attention-on")

    keys = sorted(official.keys())
    assert set(keys) == set(att_on.keys()) == set(att_off.keys())

    merged = []
    for k in keys:
        o, off, on = official[k], att_off[k], att_on[k]
        merged.append(
            {
                "filename": k,
                "n_det": o["n_det"],
                "official_abs": o["abs_rel"],
                "off_abs": off["abs_rel"],
                "on_abs": on["abs_rel"],
                "official_d1": o["delta1"],
                "off_d1": off["delta1"],
                "on_d1": on["delta1"],
                "delta_on_vs_off": on["abs_rel"] - off["abs_rel"],
                "delta_on_vs_official": on["abs_rel"] - o["abs_rel"],
                "delta_off_vs_official": off["abs_rel"] - o["abs_rel"],
            }
        )

    n = len(merged)
    win_on_vs_off = sum(1 for x in merged if x["delta_on_vs_off"] < 0)
    win_off_vs_official = sum(1 for x in merged if x["delta_off_vs_official"] < 0)
    win_on_vs_official = sum(1 for x in merged if x["delta_on_vs_official"] < 0)

    print("=" * 60)
    print(f"PER-SAMPLE ANALYSIS: {eval_root} ({n} images)")
    print("=" * 60)

    print("\nWin rate (lower abs_rel = win):")
    print(f"  Attention ON vs OFF:       {win_on_vs_off}/{n} ({100 * win_on_vs_off / n:.1f}%)")
    print(
        f"  Attention OFF vs Official:   {win_off_vs_official}/{n} "
        f"({100 * win_off_vs_official / n:.1f}%)"
    )
    print(
        f"  Attention ON vs Official:    {win_on_vs_official}/{n} "
        f"({100 * win_on_vs_official / n:.1f}%)"
    )

    print("\nMean abs_rel delta (positive = worse than baseline):")
    print(f"  ON - OFF:       {statistics.mean(x['delta_on_vs_off'] for x in merged):+.4f}")
    print(
        f"  OFF - Official: {statistics.mean(x['delta_off_vs_official'] for x in merged):+.4f}"
    )
    print(
        f"  ON - Official:  {statistics.mean(x['delta_on_vs_official'] for x in merged):+.4f}"
    )

    print("\nBy detection count:")
    for b in ["1", "2-3", "4+"]:
        sub = [x for x in merged if bucket(x["n_det"]) == b]
        cnt = len(sub)
        mean_off = statistics.mean(x["official_abs"] for x in sub)
        mean_on = statistics.mean(x["on_abs"] for x in sub)
        mean_delta = statistics.mean(x["delta_on_vs_official"] for x in sub)
        wins = sum(1 for x in sub if x["delta_on_vs_official"] < 0)
        print(
            f"  n={b} (n={cnt}): official={mean_off:.4f} on={mean_on:.4f} "
            f"delta(on-official)={mean_delta:+.4f} "
            f"win_on={wins}/{cnt} ({100 * wins / cnt:.0f}%)"
        )

    def print_top(title: str, items: list, key: str):
        print(f"\n{title}")
        for x in items:
            print(
                f"  {x[key]:+.4f}  off={x['off_abs']:.4f} on={x['on_abs']:.4f} "
                f"offi={x['official_abs']:.4f}  n={x['n_det']}  {x['filename']}"
            )

    print_top(
        "Top 10: Attention ON beats Official most (delta = on - official):",
        sorted(merged, key=lambda x: x["delta_on_vs_official"])[:10],
        "delta_on_vs_official",
    )
    print_top(
        "Top 10: Attention ON worst vs Official:",
        sorted(merged, key=lambda x: x["delta_on_vs_official"], reverse=True)[:10],
        "delta_on_vs_official",
    )
    print_top(
        "Top 10: Attention ON beats OFF:",
        sorted(merged, key=lambda x: x["delta_on_vs_off"])[:10],
        "delta_on_vs_off",
    )
    print_top(
        "Top 10: Attention ON hurts vs OFF:",
        sorted(merged, key=lambda x: x["delta_on_vs_off"], reverse=True)[:10],
        "delta_on_vs_off",
    )

    by_n: dict[int, list[float]] = defaultdict(list)
    for x in merged:
        by_n[x["n_det"]].append(x["delta_on_vs_official"])
    print("\nPer n_detections (ON - Official mean):")
    for det_n in sorted(by_n):
        vals = by_n[det_n]
        wins = sum(1 for v in vals if v < 0)
        print(
            f"  n={det_n:2d}: count={len(vals):3d} "
            f"mean_delta={statistics.mean(vals):+.4f} win={wins}/{len(vals)}"
        )

    hard = [x for x in merged if x["official_abs"] > 0.10]
    print(f"\nHard images (official abs_rel > 0.10): {len(hard)}")
    if hard:
        print(f"  mean official={statistics.mean(x['official_abs'] for x in hard):.4f}")
        print(f"  mean on={statistics.mean(x['on_abs'] for x in hard):.4f}")
        print(
            f"  mean delta(on-official)="
            f"{statistics.mean(x['delta_on_vs_official'] for x in hard):+.4f}"
        )
        wins = sum(1 for x in hard if x["delta_on_vs_official"] < 0)
        print(f"  win on vs official: {wins}/{len(hard)}")

    easy = [x for x in merged if x["official_abs"] < 0.04]
    print(f"\nEasy images (official abs_rel < 0.04): {len(easy)}")
    if easy:
        print(
            f"  mean delta(on-official)="
            f"{statistics.mean(x['delta_on_vs_official'] for x in easy):+.4f}"
        )
        wins = sum(1 for x in easy if x["delta_on_vs_official"] < 0)
        print(f"  win on vs official: {wins}/{len(easy)}")

    out = eval_root / "per_sample_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
