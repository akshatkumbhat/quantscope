"""Deterministic machine-readable diff artifact (ADR-015).

Canonical field ordering (sorted keys), path-sorted entries, no
timestamps, no absolute paths, repr-round-trip floats (identical on
Python 3.11/3.12), atomic temp-file-then-rename writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from quantscope.regression.compare import DiffReport

__all__ = ["artifact_digest", "atomic_write_json", "diff_payload", "write_diff"]

DIFF_SCHEMA_VERSION = 1


def artifact_digest(document: dict) -> str:
    """SHA-256 of the canonical (key-sorted) JSON of a document."""
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def diff_payload(report: DiffReport, *, failure_category: str) -> dict:
    """The canonical diff document (deterministic; no timestamps/paths)."""
    return {
        "diff_schema_version": DIFF_SCHEMA_VERSION,
        "baseline_name": report.baseline_name,
        "baseline_canonical_digest": report.baseline_digest,
        "artifact_digest": report.artifact_digest,
        "verdict": "pass" if report.passed else "fail",
        "failure_category": failure_category,
        "failure_counts": report.failure_counts(),
        "environment": report.environment,
        "checks": [entry.to_payload() for entry in report.entries],
    }


def write_diff(report: DiffReport, path: Path) -> None:
    category = "none" if report.passed else "regression"
    atomic_write_json(path, diff_payload(report, failure_category=category))
