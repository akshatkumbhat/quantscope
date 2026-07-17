"""Baseline capture workflow (ADR-015).

``check`` never writes baselines; capture creates a *proposed*
baseline that is still a code-review decision. Overwrites require an
explicit flag and produce a reviewable comparison against the previous
baseline. Never run in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from quantscope.regression.compare import HarnessError
from quantscope.regression.diff import artifact_digest, atomic_write_json
from quantscope.regression.models import BaselineSpec
from quantscope.regression.smoke import ARTIFACT_TYPE, build_smoke_baseline

__all__ = ["capture_baseline", "load_baseline"]


def load_baseline(path: str | Path) -> BaselineSpec:
    """Parse + validate a baseline file (schema, digest, rule uniqueness)."""
    path = Path(path)
    if not path.exists():
        raise HarnessError(f"baseline not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise HarnessError(f"baseline {path} is not valid JSON: {error}") from error
    try:
        baseline = BaselineSpec.model_validate(payload)
    except ValidationError as error:
        raise HarnessError(f"malformed baseline {path}: {error}") from error
    try:
        baseline.verify_digest()
    except ValueError as error:
        raise HarnessError(str(error)) from error
    return baseline


def capture_baseline(
    artifact: dict, output: str | Path, *, overwrite: bool = False
) -> tuple[BaselineSpec, dict | None]:
    """Create a proposed baseline from an artifact.

    Returns (baseline, comparison-with-previous-or-None). Refuses to
    overwrite without the explicit flag; validates before writing;
    writes atomically.
    """
    output = Path(output)
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise HarnessError(
            f"capture supports artifact_type {ARTIFACT_TYPE!r} in v1, got "
            f"{artifact.get('artifact_type')!r}"
        )
    previous: BaselineSpec | None = None
    if output.exists():
        if not overwrite:
            raise HarnessError(
                f"{output} exists; baseline updates are code-review decisions — pass an "
                "explicit overwrite flag to propose a replacement"
            )
        previous = load_baseline(output)

    command = f"quantscope regression capture <artifact> --output {output.name}"
    baseline = build_smoke_baseline(artifact, capture_command=command)
    baseline.verify_digest()  # validate the proposal before writing

    comparison = None
    if previous is not None:
        old_rules = {r.path: r for r in previous.rules}
        new_rules = {r.path: r for r in baseline.rules}
        changed = {
            path: {
                "previous_expected": old_rules[path].expected,
                "proposed_expected": new_rules[path].expected,
            }
            for path in sorted(set(old_rules) & set(new_rules))
            if old_rules[path].expected != new_rules[path].expected
        }
        comparison = {
            "previous_digest": previous.canonical_digest,
            "proposed_digest": baseline.canonical_digest,
            "candidate_artifact_digest": artifact_digest(artifact),
            "added_paths": sorted(set(new_rules) - set(old_rules)),
            "removed_paths": sorted(set(old_rules) - set(new_rules)),
            "changed_expectations": changed,
        }

    atomic_write_json(output, baseline.model_dump(mode="json"))
    return baseline, comparison
