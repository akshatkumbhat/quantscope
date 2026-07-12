#!/usr/bin/env python
"""Check benchmark-A acceptance conditions over the 3-seed validation runs.

Conditions (agreed 2026-07-11, incl. the documented FP32 gate deviation):
  1. Aggregate (mean) NLL and margin preserve FP32 > W8A8 > W4A4
     (NLL ascending, margin descending).
  2. Mean W4A4 accuracy is clearly degraded vs FP32 (> 1 pp drop).
  3. Mean W8A8 accuracy drop is smaller than mean W4A4 drop.
  4. Mean FP32 accuracy <= 0.96 and no seed fully saturated (>= 0.995).
  5. W8A8 effect visible in aggregate NLL and/or margin (not required
     per-seed in accuracy, which is discrete).

Exit code 0 = accept, 1 = reject (with the failed condition printed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SEEDS = (0, 1, 2)
SATURATION = 0.995


def load(run_dir: Path) -> dict[str, float]:
    metrics = json.loads((run_dir / "metrics.json").read_text())["metrics"]
    return {m["name"]: m["value"] for m in metrics}


def main(base: str = "runs/validation") -> int:
    rows = []
    for seed in SEEDS:
        run_dir = Path(base) / f"texture-a-seed{seed}-texture-a"
        if not (run_dir / "metrics.json").exists():
            print(f"MISSING: {run_dir}")
            return 1
        rows.append(load(run_dir))

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / len(rows)

    print(f"{'metric':<22}" + "".join(f"seed{s:<8}" for s in SEEDS) + "mean")
    for key in (
        "fp32_accuracy",
        "W8A8_accuracy",
        "W4A4_accuracy",
        "fp32_nll",
        "W8A8_nll",
        "W4A4_nll",
        "fp32_mean_margin",
        "W8A8_mean_margin",
        "W4A4_mean_margin",
    ):
        vals = [r[key] for r in rows]
        print(f"{key:<22}" + "".join(f"{v:<12.4f}" for v in vals) + f"{mean(key):.4f}")

    failures: list[str] = []
    if not (mean("fp32_nll") < mean("W8A8_nll") < mean("W4A4_nll")):
        failures.append("1: aggregate NLL not monotone FP32 < W8A8 < W4A4")
    if not (mean("fp32_mean_margin") > mean("W8A8_mean_margin") > mean("W4A4_mean_margin")):
        failures.append("1: aggregate margin not monotone FP32 > W8A8 > W4A4")
    w8_drop = mean("fp32_accuracy") - mean("W8A8_accuracy")
    w4_drop = mean("fp32_accuracy") - mean("W4A4_accuracy")
    if w4_drop <= 0.01:
        failures.append(f"2: mean W4A4 accuracy drop {w4_drop:.4f} not clear (> 1pp)")
    if not w8_drop < w4_drop:
        failures.append(f"3: W8A8 drop {w8_drop:.4f} not smaller than W4A4 drop {w4_drop:.4f}")
    if mean("fp32_accuracy") > 0.96 + 1e-9:
        failures.append(f"4: mean FP32 accuracy {mean('fp32_accuracy'):.4f} > 0.96")
    saturated = [s for s, r in zip(SEEDS, rows, strict=True) if r["fp32_accuracy"] >= SATURATION]
    if saturated:
        failures.append(f"4: seeds fully saturated: {saturated}")
    w8_effect = (mean("W8A8_nll") - mean("fp32_nll")) > 0 or (
        mean("fp32_mean_margin") - mean("W8A8_mean_margin")
    ) > 0
    if not w8_effect:
        failures.append("5: no W8A8 effect in aggregate NLL or margin")

    if failures:
        print("\nREJECT:")
        for f in failures:
            print(f"  - condition {f}")
        return 1
    print("\nACCEPT: all conditions met (FP32 88-94% band miss is a documented deviation)")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
