"""Numerical-regression harness (ADR-015): deterministic,
provenance-aware baseline checks with explicit tolerances."""

from quantscope.regression.capture import capture_baseline, load_baseline
from quantscope.regression.compare import (
    DiffEntry,
    DiffReport,
    HarnessError,
    check_artifact,
    resolve_pointer,
)
from quantscope.regression.diff import artifact_digest, atomic_write_json, write_diff
from quantscope.regression.models import (
    BASELINE_SCHEMA_VERSION,
    BaselineSpec,
    CheckRule,
    ComparatorType,
    EnvironmentRules,
)
from quantscope.regression.smoke import (
    build_smoke_baseline,
    generate_smoke_artifact,
    write_smoke_artifact,
)

__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "BaselineSpec",
    "CheckRule",
    "ComparatorType",
    "DiffEntry",
    "DiffReport",
    "EnvironmentRules",
    "HarnessError",
    "artifact_digest",
    "atomic_write_json",
    "build_smoke_baseline",
    "capture_baseline",
    "check_artifact",
    "generate_smoke_artifact",
    "load_baseline",
    "resolve_pointer",
    "write_diff",
    "write_smoke_artifact",
]
