"""Comparator engine and regression/harness-error classification
(ADR-015).

Exit-code semantics: 0 pass; 1 regression (numerical, structural, or
provenance); 2 harness/input error (malformed or incompatible inputs,
invalid configuration). All float comparison happens in float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from quantscope.regression.models import BaselineSpec, CheckRule, ComparatorType

__all__ = ["DiffEntry", "DiffReport", "HarnessError", "check_artifact", "resolve_pointer"]

_MISSING = object()


class HarnessError(Exception):
    """Malformed/incompatible input or invalid configuration → exit 2."""


def resolve_pointer(document: Any, pointer: str):
    """RFC 6901 resolution. Returns the value or the _MISSING sentinel."""
    if pointer == "":
        return document
    node = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if token not in node:
                return _MISSING
            node = node[token]
        elif isinstance(node, list):
            if not token.isdigit() or int(token) >= len(node):
                return _MISSING
            node = node[int(token)]
        else:
            return _MISSING
    return node


@dataclass(frozen=True)
class DiffEntry:
    path: str
    comparator: str
    passed: bool
    expected: Any
    actual: Any
    explanation: str
    atol: float | None = None
    rtol: float | None = None
    abs_diff: float | None = None
    rel_diff: float | None = None

    def to_payload(self) -> dict:
        payload = {
            "path": self.path,
            "comparator": self.comparator,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "explanation": self.explanation,
        }
        for key in ("atol", "rtol", "abs_diff", "rel_diff"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass
class DiffReport:
    baseline_name: str
    baseline_digest: str
    artifact_digest: str
    environment: dict[str, str]
    entries: list[DiffEntry] = field(default_factory=list)

    @property
    def failures(self) -> list[DiffEntry]:
        return [e for e in self.entries if not e.passed]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.failures:
            counts[entry.comparator] = counts.get(entry.comparator, 0) + 1
        return counts


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def _check_environment(artifact: dict, baseline: BaselineSpec) -> None:
    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        raise HarnessError("artifact has no environment block")
    rules = baseline.environment_rules
    for key, expected in rules.exact.items():
        actual = environment.get(key)
        _require(
            actual == expected,
            f"environment gate: {key}={actual!r}, baseline requires exactly {expected!r} "
            "(incompatible run environment, not a code regression)",
        )
    for key, allowed in rules.allowed.items():
        actual = environment.get(key)
        _require(
            actual in allowed,
            f"environment gate: {key}={actual!r} not in allowed set {allowed}",
        )


def _leaf_value(rule: CheckRule, raw: Any) -> tuple[Any, str | None]:
    """Unwrap a {value, provenance} leaf for metric rules."""
    if rule.provenance is None:
        return raw, None
    if not isinstance(raw, dict) or "value" not in raw or "provenance" not in raw:
        raise HarnessError(
            f"{rule.path}: rule expects a {{value, provenance}} leaf, artifact has "
            f"{type(raw).__name__}"
        )
    return raw["value"], raw["provenance"]


def _numeric(rule: CheckRule, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise HarnessError(f"{rule.path}: expected a number, artifact has {value!r}")
    return float(value)


def _compare_close(rule: CheckRule, actual: float) -> DiffEntry:
    expected = float(rule.expected)
    if not math.isfinite(actual):
        return DiffEntry(
            path=rule.path,
            comparator="close",
            passed=False,
            expected=expected,
            actual=actual,
            explanation="non-finite value (baseline permits none)",
            atol=rule.atol,
            rtol=rule.rtol,
        )
    abs_diff = abs(actual - expected)
    rel_diff = abs_diff / abs(expected) if expected != 0 else math.inf if abs_diff else 0.0
    passed = abs_diff <= rule.atol + rule.rtol * abs(expected)
    return DiffEntry(
        path=rule.path,
        comparator="close",
        passed=passed,
        expected=expected,
        actual=actual,
        explanation="within tolerance" if passed else "outside atol + rtol*|expected|",
        atol=rule.atol,
        rtol=rule.rtol,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
    )


def _compare_no_worse(rule: CheckRule, actual: float) -> DiffEntry:
    expected = float(rule.expected)
    if rule.direction == "higher_is_better":
        passed = actual >= expected - rule.degradation_atol
    else:
        passed = actual <= expected + rule.degradation_atol
    return DiffEntry(
        path=rule.path,
        comparator="no_worse",
        passed=passed,
        expected=expected,
        actual=actual,
        explanation=(
            "no worse than baseline (boundary inclusive)"
            if passed
            else f"worse than baseline beyond degradation_atol ({rule.direction})"
        ),
        atol=rule.degradation_atol,
        abs_diff=abs(actual - expected),
    )


def _compare_structure(rule: CheckRule, node: Any) -> list[DiffEntry]:
    entries: list[DiffEntry] = []
    if rule.required_keys is not None:
        if not isinstance(node, dict):
            return [
                DiffEntry(
                    path=rule.path,
                    comparator="structure",
                    passed=False,
                    expected="mapping",
                    actual=type(node).__name__,
                    explanation="expected a mapping",
                )
            ]
        missing = [k for k in rule.required_keys if k not in node]
        extra = [k for k in node if k not in rule.required_keys]
        passed = not missing and (rule.allow_extra_keys or not extra)
        entries.append(
            DiffEntry(
                path=rule.path,
                comparator="structure",
                passed=passed,
                expected=sorted(rule.required_keys),
                actual=sorted(node),
                explanation=(
                    "required keys present"
                    if passed
                    else f"missing={missing} extra={extra if not rule.allow_extra_keys else []}"
                ),
            )
        )
    if rule.required_length is not None:
        if not isinstance(node, list):
            entries.append(
                DiffEntry(
                    path=rule.path,
                    comparator="structure",
                    passed=False,
                    expected="sequence",
                    actual=type(node).__name__,
                    explanation="expected a sequence",
                )
            )
        else:
            ok = (
                len(node) == rule.required_length
                if not rule.allow_extra_elements
                else len(node) >= rule.required_length
            )
            entries.append(
                DiffEntry(
                    path=rule.path,
                    comparator="structure",
                    passed=ok,
                    expected=rule.required_length,
                    actual=len(node),
                    explanation="length ok" if ok else "sequence length mismatch",
                )
            )
    return entries


def check_artifact(artifact: dict, baseline: BaselineSpec) -> DiffReport:
    """Evaluate every baseline rule against the artifact.

    Raises HarnessError (exit 2) for malformed/incompatible inputs;
    returns a DiffReport whose exit_code is 0 or 1 otherwise.
    """
    _require(isinstance(artifact, dict), "artifact is not a JSON mapping")
    artifact_type = artifact.get("artifact_type")
    _require(
        artifact_type == baseline.artifact_type,
        f"incompatible artifact type {artifact_type!r}; baseline checks {baseline.artifact_type!r}",
    )
    version = artifact.get("artifact_schema_version")
    _require(
        version in baseline.compatible_artifact_schema_versions,
        f"unsupported artifact_schema_version {version!r}; baseline supports "
        f"{baseline.compatible_artifact_schema_versions}",
    )
    baseline.verify_digest()
    _check_environment(artifact, baseline)

    from quantscope.regression.diff import artifact_digest  # circular-import guard

    report = DiffReport(
        baseline_name=baseline.baseline_name,
        baseline_digest=baseline.canonical_digest,
        artifact_digest=artifact_digest(artifact),
        environment={k: str(v) for k, v in sorted(artifact.get("environment", {}).items())},
    )

    for rule in baseline.rules:
        raw = resolve_pointer(artifact, rule.path)
        if raw is _MISSING:
            # Compatible artifact missing a checked field: a regression.
            report.entries.append(
                DiffEntry(
                    path=rule.path,
                    comparator=rule.comparator.value,
                    passed=False,
                    expected=rule.expected,
                    actual=None,
                    explanation="required field missing from artifact",
                )
            )
            continue
        if rule.comparator is ComparatorType.STRUCTURE:
            report.entries.extend(_compare_structure(rule, raw))
            continue

        value, provenance = _leaf_value(rule, raw)
        if rule.provenance is not None and provenance != rule.provenance:
            report.entries.append(
                DiffEntry(
                    path=rule.path,
                    comparator="provenance",
                    passed=False,
                    expected=rule.provenance,
                    actual=provenance,
                    explanation="provenance label changed (fails regardless of value)",
                )
            )
            continue
        if rule.comparator is ComparatorType.EXACT:
            passed = value == rule.expected and type(value) is type(rule.expected)
            report.entries.append(
                DiffEntry(
                    path=rule.path,
                    comparator="exact",
                    passed=passed,
                    expected=rule.expected,
                    actual=value,
                    explanation="exact match" if passed else "exact mismatch",
                )
            )
        elif rule.comparator is ComparatorType.CLOSE:
            report.entries.append(_compare_close(rule, _numeric(rule, value)))
        elif rule.comparator is ComparatorType.NO_WORSE:
            report.entries.append(_compare_no_worse(rule, _numeric(rule, value)))
        else:  # pragma: no cover - enum is exhaustive
            raise HarnessError(f"unknown comparator {rule.comparator}")

    report.entries.sort(key=lambda e: e.path)  # deterministic ordering
    return report
