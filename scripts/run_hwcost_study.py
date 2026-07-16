#!/usr/bin/env python
"""ADR-014: enrich the B3 sweep artifacts with analytical hardware costs.

Reads the frozen B3 sweep tables (never modifying them), computes
transparent component-wise ESTIMATED costs for all 256 configurations
per checkpoint under the schema-v1 generic_edge_npu profile, recomputes
the exact NLL-vs-estimated-cost Pareto frontier, generates
recommendations at the preregistered normalized budgets 0.60/0.75/0.90,
and compares against the historical weight-bits proxy (which stays in
the original artifacts, unrewritten).

Provenance: quality metrics are quoted SIMULATED values from B3; every
cost is ESTIMATED for the modeled quantizable workload only (omitted
constant costs can overstate the fraction of system-level savings);
recommendations are derived from simulated quality and estimated cost.
No measured performance claim anywhere.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from scipy.stats import spearmanr

from quantscope.benchmark import benchmark_config
from quantscope.config import Provenance
from quantscope.hardware import (
    GROUP_ORDER_V1,
    GROUP_ORDER_VERSION,
    account_model,
    config_identifier,
    configuration_cost,
    load_hardware_profile,
    recommend_for_budget,
)
from quantscope.hardware.profile import SUPPORTED_SCHEMA_VERSION
from quantscope.models.tiny_cnn import build_model
from quantscope.reporting import load_sweep_records
from quantscope.search import pareto_frontier
from quantscope.search.exhaustive import SweepRecord
from quantscope.utilities import RunWriter

SEEDS = (0, 1, 2)
VALIDATION_DIR = Path("runs/validation-012")
SUMMARY_PATH = VALIDATION_DIR / "hwcost-study-summary.json"
PROFILE_PATH = Path("configs/hardware/generic_edge_npu.yaml")
BUDGETS = (0.60, 0.75, 0.90)  # normalized modeled-workload cost
SYSTEM_WARNING = (
    "costs cover the modeled quantizable workload only; omitted constant costs "
    "(BatchNorm, adds, pooling, flatten, logits, quant/dequant boundaries) can "
    "overstate the fraction of system-level savings"
)


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write to a temp file in the target directory, rename atomically."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _enrich_records(records, accounting, profile):
    """Component-wise costs for every configuration (validated first)."""
    # Validate every assignment BEFORE any artifact is written.
    for record in records:
        if len(record.bits) != len(GROUP_ORDER_V1):
            raise ValueError(
                f"sweep record has {len(record.bits)} groups; expected {len(GROUP_ORDER_V1)}"
            )
        for b in record.bits:
            profile.compute_coefficient(b, b)  # raises on unsupported pairs

    all_int8 = configuration_cost(accounting, [(8, 8)] * len(GROUP_ORDER_V1), profile)
    all_int4 = configuration_cost(accounting, [(4, 4)] * len(GROUP_ORDER_V1), profile)
    baseline_total = all_int8.total

    enriched = []
    for record in records:
        assignment = [(b, b) for b in record.bits]
        cost = configuration_cost(accounting, assignment, profile)
        normalized = cost.total / baseline_total
        enriched.append(
            {
                "identifier": cost.identifier,
                "bits": list(record.bits),
                "nll": record.nll,  # simulated (quoted from B3)
                "accuracy": record.accuracy,  # simulated (quoted from B3)
                "old_proxy_cost": record.cost,  # historical baseline, estimated
                "components_ncu": cost.components.to_dict(),  # estimated
                "per_group_ncu": {g: c.to_dict() for g, c in cost.per_group.items()},
                "raw_total_ncu": cost.total,
                "normalized_cost": normalized,  # estimated
            }
        )

    int4_norm = all_int4.total / baseline_total
    int8_row = next(e for e in enriched if all(b == 8 for b in e["bits"]))
    if not math.isclose(int8_row["normalized_cost"], 1.0, abs_tol=1e-12):
        raise AssertionError("all-INT8 normalized cost is not 1.0")
    for entry in enriched:
        if not (int4_norm - 1e-12 <= entry["normalized_cost"] <= 1.0 + 1e-12):
            raise AssertionError(
                f"{entry['identifier']}: normalized cost {entry['normalized_cost']} outside "
                f"[{int4_norm}, 1.0]"
            )
    return enriched, int4_norm


def _proxy_comparison(enriched, budgets):
    """Historical weight-bits proxy vs the new analytical model."""
    old = [e["old_proxy_cost"] for e in enriched]
    new = [e["normalized_cost"] for e in enriched]
    rho = float(spearmanr(old, new).statistic)

    def frontier_ids(cost_key):
        records = [
            SweepRecord(
                bits=tuple(e["bits"]), cost=e[cost_key], nll=e["nll"], accuracy=e["accuracy"]
            )
            for e in enriched
        ]
        return {r.bits for r in pareto_frontier(records, quality="nll")}

    new_frontier = frontier_ids("normalized_cost")
    old_frontier = frontier_ids("old_proxy_cost")
    jaccard = len(new_frontier & old_frontier) / len(new_frontier | old_frontier)

    old_records = [dict(e, normalized_cost=e["old_proxy_cost"]) for e in enriched]
    recommendation_changes = {}
    for budget in budgets:
        new_rec = recommend_for_budget(enriched, budget)
        old_rec = recommend_for_budget(old_records, budget)
        new_id = new_rec["recommendation"]["identifier"] if new_rec["feasible"] else None
        old_id = old_rec["recommendation"]["identifier"] if old_rec["feasible"] else None
        recommendation_changes[f"budget_{budget:.2f}"] = {
            "new_model": new_id,
            "old_proxy": old_id,
            "changed": new_id != old_id,
        }

    # Concrete ordering reversal: the old proxy ties configurations with
    # equal weight-bit totals; the new model separates them via compute
    # and activation terms. Report the widest such separation.
    by_old: dict[float, list[dict]] = {}
    for e in enriched:
        by_old.setdefault(e["old_proxy_cost"], []).append(e)
    example = None
    best_spread = 0.0
    for tied in by_old.values():
        if len(tied) < 2:
            continue
        lo = min(tied, key=lambda e: e["normalized_cost"])
        hi = max(tied, key=lambda e: e["normalized_cost"])
        spread = hi["normalized_cost"] - lo["normalized_cost"]
        if spread > best_spread:
            best_spread = spread
            example = {
                "old_proxy_cost_tied_at": lo["old_proxy_cost"],
                "cheaper_under_new_model": {
                    "identifier": lo["identifier"],
                    "normalized_cost": lo["normalized_cost"],
                    "components_ncu": lo["components_ncu"],
                },
                "costlier_under_new_model": {
                    "identifier": hi["identifier"],
                    "normalized_cost": hi["normalized_cost"],
                    "components_ncu": hi["components_ncu"],
                },
                "explanation": "equal weight-storage bits; activation-memory and "
                "compute terms (absent from the proxy) separate the pair",
            }
    return {
        "spearman_rho_old_vs_new": rho,
        "pareto_membership": {
            "new_frontier_size": len(new_frontier),
            "old_frontier_size": len(old_frontier),
            "shared": len(new_frontier & old_frontier),
            "jaccard": jaccard,
        },
        "recommendation_changes": recommendation_changes,
        "ordering_reversal_example": example,
    }


def main() -> int:
    if SUMMARY_PATH.exists():
        print(f"REFUSED: {SUMMARY_PATH} exists — the enrichment study runs once.")
        return 2

    loaded = load_hardware_profile(PROFILE_PATH)
    accounting = account_model(build_model(benchmark_config(seed=0).model))
    accounting_digest = accounting.digest()
    print(f"profile digest: {loaded.canonical_digest}")
    print(f"accounting digest: {accounting_digest}")

    summary: dict = {
        "design": "ADR-014 + addendum; analytical hardware cost model",
        "provenance": {
            "quality_metrics": "simulated (quoted from B3; originals unmodified)",
            "cost_components": "estimated (fictional profile; modeled workload only)",
            "recommendations": "derived from simulated quality and estimated cost",
        },
        "profile": {
            "path": str(PROFILE_PATH),
            "source_sha256": loaded.source_sha256,
            "canonical_digest": loaded.canonical_digest,
            "schema_version": SUPPORTED_SCHEMA_VERSION,
        },
        "accounting_digest": accounting_digest,
        "group_order_version": GROUP_ORDER_VERSION,
        "traffic_model": accounting.traffic_model,
        "system_level_warning": SYSTEM_WARNING,
        "per_seed": {},
    }

    for seed in SEEDS:
        sweep_path = VALIDATION_DIR / f"texture-a-seed{seed}-sweep" / "sweep_table.json"
        records = load_sweep_records(VALIDATION_DIR, seed)
        enriched, int4_norm = _enrich_records(records, accounting, loaded.profile)
        recommendations = [recommend_for_budget(enriched, b) for b in BUDGETS]
        frontier = pareto_frontier(
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
        comparison = _proxy_comparison(enriched, BUDGETS)

        config = benchmark_config(seed=seed, output_dir=str(VALIDATION_DIR), freq_step=0.12)
        writer = RunWriter(config, kind="hwcost")
        _atomic_write_json(writer.run_dir / "hwcost_table.json", enriched)
        writer.record_metric(
            "lineage",
            {
                "source_b3_artifact": str(sweep_path),
                "source_b3_sha256": hashlib.sha256(sweep_path.read_bytes()).hexdigest(),
                "checkpoint": f"texture-a-seed{seed} (freq_step=0.12)",
                "profile_path": str(PROFILE_PATH),
                "profile_source_sha256": loaded.source_sha256,
                "profile_canonical_digest": loaded.canonical_digest,
                "accounting_digest": accounting_digest,
                "group_order_version": GROUP_ORDER_VERSION,
                "cost_model_schema_version": SUPPORTED_SCHEMA_VERSION,
                "traffic_model": accounting.traffic_model,
                "excluded_operations": accounting.excluded_operations,
                "modeled_weighted_modules": accounting.modeled_weighted_modules,
                "modeled_total_macs": accounting.total_macs,
                "modeled_total_parameters": accounting.total_parameters,
                "system_level_warning": SYSTEM_WARNING,
            },
            Provenance.MEASURED,
            note="artifact lineage and accounting facts",
        )
        writer.record_metric(
            "all_int4_normalized_cost",
            int4_norm,
            Provenance.ESTIMATED,
            note="modeled quantizable workload; fictional profile",
        )
        writer.record_metric(
            "pareto_frontier",
            [config_identifier([(b, b) for b in r.bits]) for r in frontier],
            Provenance.ESTIMATED,
            note="NLL (simulated) vs normalized estimated cost",
        )
        writer.record_metric(
            "recommendations",
            recommendations,
            Provenance.ESTIMATED,
            note="derived from simulated quality and estimated cost; budgets on "
            "normalized modeled-workload cost; never relaxed",
        )
        writer.record_metric(
            "proxy_comparison",
            comparison,
            Provenance.ESTIMATED,
            note="historical weight-bits proxy retained in original artifacts",
        )
        print(f"artifact: {writer.finalize()}")

        summary["per_seed"][f"seed{seed}"] = {
            "all_int4_normalized_cost": int4_norm,
            "pareto_frontier_size": len(frontier),
            "recommendations": recommendations,
            "proxy_comparison": comparison,
        }

    _atomic_write_json(SUMMARY_PATH, summary)
    print(f"\nsummary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
