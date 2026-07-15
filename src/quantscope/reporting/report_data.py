"""Artifact loading for the reporting phase (read-only, fail-loud).

The generated run artifacts are the numerical source of truth for every
report figure (ADR-012 addendum 6). Loaders here never modify, rerun,
or recompute benchmark results; they read the labeled metrics exactly
as persisted and raise actionable errors when a required artifact or
field is missing, instead of silently plotting partial data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from quantscope.search import SweepRecord

__all__ = [
    "file_sha256",
    "get_metric",
    "load_labeled_metrics",
    "load_observer_summary",
    "load_sweep_records",
]


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a source artifact, for the figure manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required report artifact missing: {path}")
    return json.loads(path.read_text())


def load_labeled_metrics(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load a run's ``metrics.json`` keyed by metric name.

    Every entry keeps its provenance label; an artifact entry without a
    provenance label is a corrupt artifact and raises.
    """
    path = Path(run_dir) / "metrics.json"
    payload = _read_json(path)
    entries: dict[str, dict[str, Any]] = {}
    for entry in payload.get("metrics", []):
        if "provenance" not in entry:
            raise ValueError(f"metric {entry.get('name')!r} in {path} has no provenance label")
        entries[entry["name"]] = entry
    if not entries:
        raise ValueError(f"no metrics found in {path}")
    return entries


def get_metric(entries: dict[str, dict[str, Any]], name: str, *, source: str = "") -> Any:
    """A metric's value, or an actionable error naming the missing field."""
    if name not in entries:
        where = f" in {source}" if source else ""
        raise KeyError(f"required metric {name!r} missing{where}")
    return entries[name]["value"]


def load_sweep_records(base: str | Path, seed: int) -> list[SweepRecord]:
    """One checkpoint's exhaustive B3 sweep table (256 configurations)."""
    path = Path(base) / f"texture-a-seed{seed}-sweep" / "sweep_table.json"
    rows = _read_json(path)
    records = []
    for row in rows:
        missing = [k for k in ("bits", "cost", "nll", "accuracy") if k not in row]
        if missing:
            raise ValueError(f"sweep record in {path} missing fields: {missing}")
        records.append(
            SweepRecord(
                bits=tuple(row["bits"]), cost=row["cost"], nll=row["nll"], accuracy=row["accuracy"]
            )
        )
    if not records:
        raise ValueError(f"empty sweep table: {path}")
    return records


def load_observer_summary(path: str | Path) -> dict[str, Any]:
    """The cross-seed D observer-study summary artifact."""
    summary = _read_json(Path(path))
    required = (
        "Q1_robustness_primary_condition",
        "Q2_clean_non_inferiority",
        "Q3_pow2_cost_measurement_only",
        "mechanism_decomposition",
    )
    missing = [k for k in required if k not in summary]
    if missing:
        raise ValueError(f"observer-study summary {path} missing sections: {missing}")
    return summary
