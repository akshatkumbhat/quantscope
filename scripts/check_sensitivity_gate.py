#!/usr/bin/env python
"""Evaluate the plan-step-B stop gate over the 3-seed W4A4 group ablations.

Gate (agreed 2026-07-12, ADR-008 addendum 2):
  CONTINUE if per-group results show meaningful, non-uniform sensitivity
  in ΔNLL, prediction flips, or W4A4 accuracy, stable across seeds.
  STOP before plan step C if the ranking is effectively flat, dominated
  by ties, or unstable across seeds.

Operationalization (thresholds documented here, not tuned post hoc):
  1. Meaningful: max mean ΔNLL > 0.005, or max mean flip rate > 0.005.
  2. Non-uniform: max/median of mean ΔNLL >= 3 (review-suggested ratio).
  3. Stable: pairwise Spearman of ΔNLL rankings >= 0.7 for at least 2 of
     3 seed pairs, OR the top-2 groups (by mean ΔNLL) are top-2 in at
     least 2 of 3 individual seeds.

Exit code 0 = CONTINUE, 1 = STOP (with failed criteria printed).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

SEEDS = (0, 1, 2)


def load_deltas(run_dir: Path) -> dict[str, dict[str, float]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())["metrics"]
    groups: dict[str, dict[str, float]] = {}
    for m in metrics:
        name = m["name"]
        for suffix in ("delta_nll", "prediction_flip_rate", "delta_accuracy", "delta_margin"):
            if name.endswith(suffix):
                group = name[: -(len(suffix) + 1)]
                groups.setdefault(group, {})[suffix] = m["value"]
    return groups


def main(base: str = "runs/validation") -> int:
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        run_dir = Path(base) / f"texture-a-seed{seed}-ablation-w4a4"
        if not (run_dir / "metrics.json").exists():
            print(f"MISSING: {run_dir}")
            return 1
        per_seed[seed] = load_deltas(run_dir)

    group_names = sorted(per_seed[SEEDS[0]])
    mean_dnll = {
        g: sum(per_seed[s][g]["delta_nll"] for s in SEEDS) / len(SEEDS) for g in group_names
    }
    mean_flips = {
        g: sum(per_seed[s][g]["prediction_flip_rate"] for s in SEEDS) / len(SEEDS)
        for g in group_names
    }

    print(f"{'group':<16}{'mean dNLL':<12}{'mean flips':<12}per-seed dNLL")
    for g in sorted(group_names, key=lambda g: mean_dnll[g], reverse=True):
        per = "  ".join(f"{per_seed[s][g]['delta_nll']:+.4f}" for s in SEEDS)
        print(f"{g:<16}{mean_dnll[g]:<+12.4f}{mean_flips[g]:<12.4f}{per}")

    failures: list[str] = []

    # 1. Meaningful
    if max(mean_dnll.values()) <= 0.005 and max(mean_flips.values()) <= 0.005:
        failures.append("1: no meaningful sensitivity (max mean dNLL and flips <= 0.005)")

    # 2. Non-uniform (max/median of mean dNLL, clamped at a tiny floor)
    ordered = sorted(mean_dnll.values())
    median = (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    ratio = max(mean_dnll.values()) / max(median, 1e-6)
    print(f"\nmax/median mean-dNLL ratio: {ratio:.2f} (median {median:+.5f})")
    if ratio < 3.0:
        failures.append(f"2: ranking too uniform (max/median {ratio:.2f} < 3)")

    # 3. Stability
    rankings = {s: [per_seed[s][g]["delta_nll"] for g in group_names] for s in SEEDS}
    corrs = []
    for a, b in itertools.combinations(SEEDS, 2):
        rho = float(spearmanr(rankings[a], rankings[b]).statistic)
        corrs.append(rho)
        print(f"spearman seeds {a}-{b}: {rho:.3f}")
    strong_pairs = sum(rho >= 0.7 for rho in corrs)

    mean_top2 = set(sorted(group_names, key=lambda g: mean_dnll[g], reverse=True)[:2])
    top2_hits = 0
    for s in SEEDS:
        seed_top2 = set(
            sorted(group_names, key=lambda g: per_seed[s][g]["delta_nll"], reverse=True)[:2]
        )
        if seed_top2 == mean_top2:
            top2_hits += 1
    print(f"strong spearman pairs: {strong_pairs}/3; seeds matching mean top-2: {top2_hits}/3")
    if strong_pairs < 2 and top2_hits < 2:
        failures.append(
            f"3: unstable ranking (strong pairs {strong_pairs}/3, top-2 stable {top2_hits}/3)"
        )

    if failures:
        print("\nSTOP before plan step C:")
        for f in failures:
            print(f"  - criterion {f}")
        return 1
    print("\nCONTINUE: sensitivity is meaningful, non-uniform, and stable across seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
