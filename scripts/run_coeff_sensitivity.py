#!/usr/bin/env python
"""ADR-016 Part D: hardware-profile coefficient sensitivity.

Scales each profile coefficient by 0.5x and 1.5x, one at a time (12
variants), rebuilds costs on the frozen B3 tables, and reports per
checkpoint whether each budget recommendation changes and the
Pareto-membership Jaccard vs the canonical profile. Everything
ESTIMATED under a fictional profile; this quantifies how sensitive the
ADR-014 recommendations are to the assumed coefficients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from quantscope.benchmark import benchmark_config
from quantscope.hardware import account_model, configuration_cost
from quantscope.hardware.profile import HardwareProfile
from quantscope.models.tiny_cnn import build_model
from quantscope.regression import atomic_write_json
from quantscope.reporting import load_sweep_records
from quantscope.search import pareto_frontier
from quantscope.search.exhaustive import SweepRecord

SEEDS = (0, 1, 2)
VALIDATION_DIR = Path("runs/validation-012")
SUMMARY_PATH = VALIDATION_DIR / "hwcost-sensitivity-summary.json"
PROFILE_PATH = Path("configs/hardware/generic_edge_npu.yaml")
BUDGETS = (0.60, 0.75, 0.90)
FACTORS = (0.5, 1.5)


def _variants(payload: dict):
    for i, entry in enumerate(payload["compute_ncu_per_mac"]):
        for factor in FACTORS:
            changed = yaml.safe_load(yaml.safe_dump(payload))
            changed["compute_ncu_per_mac"][i]["ncu"] = entry["ncu"] * factor
            yield f"compute[W{entry['weight_bits']}A{entry['activation_bits']}]x{factor}", changed
    for key in ("weight_memory_ncu_per_bit", "activation_memory_ncu_per_bit"):
        for factor in FACTORS:
            changed = yaml.safe_load(yaml.safe_dump(payload))
            changed[key] = payload[key] * factor
            yield f"{key}x{factor}", changed


def _analysis(records, accounting, profile: HardwareProfile):
    all8 = configuration_cost(accounting, [(8, 8)] * 8, profile).total
    enriched = []
    for record in records:
        cost = configuration_cost(accounting, [(b, b) for b in record.bits], profile)
        enriched.append(
            {
                "bits": record.bits,
                "nll": record.nll,
                "accuracy": record.accuracy,
                "normalized_cost": cost.total / all8,
            }
        )
    frontier = {
        r.bits
        for r in pareto_frontier(
            [
                SweepRecord(
                    bits=tuple(e["bits"]),
                    cost=e["normalized_cost"],
                    nll=e["nll"],
                    accuracy=e["accuracy"],
                )
                for e in enriched
            ],
            quality="nll",
        )
    }
    recs = {}
    for budget in BUDGETS:
        feasible = [e for e in enriched if e["normalized_cost"] <= budget]
        recs[budget] = (
            min(feasible, key=lambda e: (e["nll"], e["normalized_cost"]))["bits"]
            if feasible
            else None
        )
    return frontier, recs


def main() -> int:
    if SUMMARY_PATH.exists():
        print(f"REFUSED: {SUMMARY_PATH} exists — the sensitivity sweep runs once.")
        return 2
    payload = yaml.safe_load(PROFILE_PATH.read_text())
    canonical = HardwareProfile.model_validate(payload)
    accounting = account_model(build_model(benchmark_config(seed=0).model))
    sweeps = {s: load_sweep_records(VALIDATION_DIR, s) for s in SEEDS}
    base = {s: _analysis(sweeps[s], accounting, canonical) for s in SEEDS}

    variants_out = {}
    for name, changed_payload in _variants(payload):
        profile = HardwareProfile.model_validate(changed_payload)
        per_seed = {}
        for s in SEEDS:
            frontier, recs = _analysis(sweeps[s], accounting, profile)
            base_frontier, base_recs = base[s]
            per_seed[f"seed{s}"] = {
                "pareto_jaccard_vs_canonical": len(frontier & base_frontier)
                / len(frontier | base_frontier),
                "recommendation_changed": {
                    f"budget_{b:.2f}": recs[b] != base_recs[b] for b in BUDGETS
                },
            }
        changed_cells = sum(
            v["recommendation_changed"][f"budget_{b:.2f}"]
            for v in per_seed.values()
            for b in BUDGETS
        )
        variants_out[name] = {"per_seed": per_seed, "changed_recommendation_cells": changed_cells}
        print(f"{name:<40} changed cells {changed_cells}/9")

    summary = {
        "design": "ADR-016 Part D: one-at-a-time +/-50% coefficient sweep (12 variants)",
        "provenance": "all costs estimated; fictional profile; quality metrics simulated "
        "(quoted from B3)",
        "budgets": list(BUDGETS),
        "variants": variants_out,
    }
    atomic_write_json(SUMMARY_PATH, summary)
    print(f"summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
