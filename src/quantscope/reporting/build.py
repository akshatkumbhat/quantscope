"""Report build orchestration: artifacts in, figures + manifest out.

Reads existing run artifacts (never modifying or recomputing them),
renders the four report figures deterministically, and writes a
machine-readable ``manifest.json`` recording, per figure: the output
path and its SHA-256, every source artifact path with its SHA-256, and
the field-level provenance labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quantscope.reporting.figures import (
    fig_mechanism_decomposition,
    fig_observer_factorial,
    fig_pareto_frontiers,
    fig_pow2_cost,
)
from quantscope.reporting.report_data import (
    file_sha256,
    get_metric,
    load_labeled_metrics,
    load_observer_summary,
    load_sweep_records,
)

__all__ = ["build_report"]

SEEDS = (0, 1, 2)
OBSERVERS = ("minmax", "percentile", "mse_grid", "pow2")


def _source(path: str | Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def build_report(
    validation_dir: str | Path,
    summary_path: str | Path,
    out_dir: str | Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> Path:
    """Build every report figure and the manifest; returns the manifest path."""
    validation_dir = Path(validation_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    # B3: per-checkpoint Pareto frontiers.
    sweeps = {s: load_sweep_records(validation_dir, s) for s in seeds}
    out = out_dir / "pareto_frontiers.png"
    entry = fig_pareto_frontiers(sweeps, out)
    entry["sources"] = [
        _source(validation_dir / f"texture-a-seed{s}-sweep" / "sweep_table.json") for s in seeds
    ]
    entries.append(entry | {"output": out.name})

    # D: per-seed factorial values straight from the labeled artifacts.
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    fp32_clean: dict[int, float] = {}
    study_sources = []
    for s in seeds:
        run_dir = validation_dir / f"texture-a-seed{s}-observer-study"
        metrics = load_labeled_metrics(run_dir)
        study_sources.append(_source(run_dir / "metrics.json"))
        fp32_clean[s] = get_metric(metrics, "fp32[clean_eval][nll]", source=str(run_dir))
        per_seed[s] = {
            obs: {
                cond: get_metric(metrics, f"{obs}|W4A4|{cond}", source=str(run_dir))["nll"]
                for cond in ("clean->clean", "stressed->clean")
            }
            for obs in OBSERVERS
        }
    out = out_dir / "observer_factorial_w4a4.png"
    entry = fig_observer_factorial(per_seed, fp32_clean, out)
    entry["sources"] = study_sources
    entries.append(entry | {"output": out.name})

    # D: mechanism decomposition and Q3, from the summary artifact.
    summary = load_observer_summary(summary_path)
    summary_source = _source(summary_path)

    out = out_dir / "mechanism_decomposition.png"
    entry = fig_mechanism_decomposition(summary["mechanism_decomposition"], out)
    entry["sources"] = [summary_source]
    entries.append(entry | {"output": out.name})

    out = out_dir / "pow2_cost.png"
    entry = fig_pow2_cost(summary["Q3_pow2_cost_measurement_only"], out)
    entry["sources"] = [summary_source]
    entries.append(entry | {"output": out.name})

    for entry in entries:
        entry["output_sha256"] = file_sha256(out_dir / entry["output"])

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "provenance_legend": {
                    "simulated": "fake-quant simulation policy v1; not integer execution",
                    "estimated": "analytical hardware cost model; not a measurement",
                    "measured": "actually executed/observed (FP32 eval; real INT8 in step C)",
                },
                "figures": entries,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest_path
