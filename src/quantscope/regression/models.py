"""Baseline and diff schemas for the numerical-regression harness
(ADR-015). Unknown fields are forbidden everywhere; a baseline is a
reviewed, committed specification — never updated implicitly."""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "BaselineSpec",
    "CheckRule",
    "ComparatorType",
    "EnvironmentRules",
]

BASELINE_SCHEMA_VERSION = 1


class ComparatorType(StrEnum):
    EXACT = "exact"
    CLOSE = "close"
    NO_WORSE = "no_worse"
    STRUCTURE = "structure"


class CheckRule(BaseModel):
    """One checked path: comparator, expectation, provenance, rationale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str  # JSON Pointer (RFC 6901)
    comparator: ComparatorType
    expected: Any = None
    # Required for metric rules (leaves shaped {value, provenance});
    # omitted only for plain identity values (identifiers, digests).
    provenance: str | None = None
    # close:
    atol: float = 0.0
    rtol: float = 0.0
    # no_worse:
    direction: Literal["higher_is_better", "lower_is_better"] | None = None
    degradation_atol: float = 0.0
    # structure:
    required_keys: list[str] | None = None
    allow_extra_keys: bool = False
    required_length: int | None = None
    allow_extra_elements: bool = False
    # Required whenever any tolerance is nonzero:
    rationale: str | None = None

    @field_validator("path")
    @classmethod
    def _pointer_syntax(cls, value: str) -> str:
        if value != "" and not value.startswith("/"):
            raise ValueError(f"path must be a JSON Pointer starting with '/': {value!r}")
        return value

    @field_validator("atol", "rtol", "degradation_atol")
    @classmethod
    def _finite_nonnegative(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"tolerance must be finite and >= 0, got {value}")
        return value

    @model_validator(mode="after")
    def _comparator_consistency(self) -> CheckRule:
        if self.comparator is ComparatorType.CLOSE:
            if not isinstance(self.expected, int | float) or isinstance(self.expected, bool):
                raise ValueError(f"{self.path}: close comparator needs a numeric expected value")
            if not math.isfinite(float(self.expected)):
                raise ValueError(f"{self.path}: expected value must be finite")
        if self.comparator is ComparatorType.NO_WORSE:
            if self.direction is None:
                raise ValueError(f"{self.path}: no_worse comparator requires a direction")
            if not isinstance(self.expected, int | float) or isinstance(self.expected, bool):
                raise ValueError(f"{self.path}: no_worse comparator needs a numeric expected")
        if self.comparator is ComparatorType.STRUCTURE:
            if self.required_keys is None and self.required_length is None:
                raise ValueError(
                    f"{self.path}: structure comparator requires required_keys and/or "
                    "required_length"
                )
        else:
            if self.required_keys is not None or self.required_length is not None:
                raise ValueError(f"{self.path}: structural fields on a non-structure rule")
        if self.comparator in (ComparatorType.EXACT, ComparatorType.STRUCTURE) and (
            self.atol or self.rtol or self.degradation_atol
        ):
            raise ValueError(f"{self.path}: tolerances are invalid for {self.comparator}")
        if (self.atol or self.rtol or self.degradation_atol) and not self.rationale:
            raise ValueError(f"{self.path}: every nonzero tolerance requires a rationale")
        return self


class EnvironmentRules(BaseModel):
    """Three-tier environment compatibility rules (ADR-015)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exact: dict[str, str] = {}
    allowed: dict[str, list[str]] = {}
    report_only: list[str] = []


class BaselineSpec(BaseModel):
    """A committed, reviewed regression baseline (schema v1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_schema_version: int
    baseline_name: str
    description: str
    artifact_type: str
    compatible_artifact_schema_versions: list[int]
    capture_command: str
    quantscope_commit: str | None = None  # traceability only; never a gate
    environment_rules: EnvironmentRules
    ignored_paths: list[str] = []
    rules: list[CheckRule]
    canonical_digest: str

    @field_validator("baseline_schema_version")
    @classmethod
    def _supported(cls, value: int) -> int:
        if value != BASELINE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported baseline_schema_version {value}; this harness supports "
                f"{BASELINE_SCHEMA_VERSION}"
            )
        return value

    @model_validator(mode="after")
    def _rules_unambiguous(self) -> BaselineSpec:
        if not self.rules:
            raise ValueError("baseline declares no rules")
        seen: set[str] = set()
        for rule in self.rules:
            if rule.path in seen:
                raise ValueError(f"duplicate rule for path {rule.path!r} (no precedence in v1)")
            seen.add(rule.path)
        overlap = seen & set(self.ignored_paths)
        if overlap:
            raise ValueError(f"paths both checked and ignored: {sorted(overlap)}")
        return self

    def body_digest(self) -> str:
        """Canonical digest over everything except the digest field."""
        body = self.model_dump(mode="json")
        body.pop("canonical_digest")
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def verify_digest(self) -> None:
        actual = self.body_digest()
        if actual != self.canonical_digest:
            raise ValueError(
                "baseline canonical_digest mismatch: baseline edited without re-capture "
                f"(recorded {self.canonical_digest[:12]}…, computed {actual[:12]}…)"
            )
