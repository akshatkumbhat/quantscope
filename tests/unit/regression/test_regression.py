"""ADR-015 regression-harness tests: schema, comparators,
classification, determinism, capture protection, and the smoke
round trip with a deliberate perturbation."""

import copy
import json
from pathlib import Path

import pytest

from quantscope.regression import (
    BaselineSpec,
    CheckRule,
    ComparatorType,
    EnvironmentRules,
    HarnessError,
    build_smoke_baseline,
    capture_baseline,
    check_artifact,
    generate_smoke_artifact,
    load_baseline,
    resolve_pointer,
    write_diff,
)
from quantscope.regression.diff import diff_payload

ENV = {
    "python": "3.11.13",
    "python_minor": "3.11",
    "torch": "2.2.2",
    "numpy": "1.26.4",
    "quantscope": "0.1.0",
    "platform": "test",
}


def _artifact(**overrides) -> dict:
    artifact = {
        "artifact_type": "unit-test",
        "artifact_schema_version": 1,
        "environment": dict(ENV),
        "sections": {
            "metrics": {
                "nll": {"value": 0.5, "provenance": "simulated"},
                "count": {"value": 64, "provenance": "measured"},
                "accuracy": {"value": 0.75, "provenance": "simulated"},
            },
            "name": "fixture",
        },
    }
    for path, value in overrides.items():
        node = artifact
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return artifact


def _baseline(rules: list[CheckRule], **overrides) -> BaselineSpec:
    payload = {
        "baseline_schema_version": 1,
        "baseline_name": "unit",
        "description": "fixture",
        "artifact_type": "unit-test",
        "compatible_artifact_schema_versions": [1],
        "capture_command": "n/a",
        "environment_rules": EnvironmentRules(
            exact={"torch": "2.2.2"}, allowed={"python_minor": ["3.11", "3.12"]}
        ),
        "rules": rules,
        "canonical_digest": "0" * 64,
    }
    payload.update(overrides)
    spec = BaselineSpec.model_validate(payload)
    return spec.model_copy(update={"canonical_digest": spec.body_digest()})


def _nll_rule(**kwargs) -> CheckRule:
    defaults = {
        "path": "/sections/metrics/nll",
        "comparator": ComparatorType.CLOSE,
        "expected": 0.5,
        "provenance": "simulated",
        "atol": 1e-6,
        "rtol": 1e-5,
        "rationale": "test tolerance",
    }
    defaults.update(kwargs)
    return CheckRule.model_validate(defaults)


class TestSchema:
    def test_valid_baseline_round_trip(self, tmp_path: Path) -> None:
        spec = _baseline([_nll_rule()])
        path = tmp_path / "b.json"
        path.write_text(json.dumps(spec.model_dump(mode="json")))
        assert load_baseline(path).canonical_digest == spec.canonical_digest

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(Exception, match="mystery"):
            _baseline([_nll_rule()], mystery=1)

    def test_duplicate_rule_rejected(self) -> None:
        with pytest.raises(Exception, match="duplicate rule"):
            _baseline([_nll_rule(), _nll_rule()])

    def test_checked_and_ignored_overlap_rejected(self) -> None:
        with pytest.raises(Exception, match="both checked and ignored"):
            _baseline([_nll_rule()], ignored_paths=["/sections/metrics/nll"])

    def test_nonzero_tolerance_requires_rationale(self) -> None:
        with pytest.raises(Exception, match="rationale"):
            _nll_rule(rationale=None)

    def test_malformed_baseline_is_harness_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(HarnessError, match="not valid JSON"):
            load_baseline(path)

    def test_edited_baseline_digest_mismatch(self, tmp_path: Path) -> None:
        spec = _baseline([_nll_rule()])
        payload = spec.model_dump(mode="json")
        payload["description"] = "edited after capture"
        path = tmp_path / "b.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(HarnessError, match="canonical_digest mismatch"):
            load_baseline(path)


class TestPointer:
    def test_mapping_sequence_and_escapes(self) -> None:
        doc = {"a": [{"x~y": 1, "p/q": 2}]}
        assert resolve_pointer(doc, "/a/0/x~0y") == 1
        assert resolve_pointer(doc, "/a/0/p~1q") == 2
        assert resolve_pointer(doc, "") == doc


class TestComparators:
    def test_exact_pass_and_fail(self) -> None:
        rule = CheckRule(
            path="/sections/metrics/count",
            comparator=ComparatorType.EXACT,
            expected=64,
            provenance="measured",
        )
        assert check_artifact(_artifact(), _baseline([rule])).passed
        bad = _artifact()
        bad["sections"]["metrics"]["count"]["value"] = 63
        report = check_artifact(bad, _baseline([rule]))
        assert not report.passed and report.exit_code == 1

    def test_atol_boundary_inclusive(self) -> None:
        # Dyadic values so the boundary is exactly representable.
        rule = _nll_rule(atol=0.015625, rtol=0.0)
        at_boundary = _artifact()
        at_boundary["sections"]["metrics"]["nll"]["value"] = 0.515625  # exactly atol away
        assert check_artifact(at_boundary, _baseline([rule])).passed
        beyond = _artifact()
        beyond["sections"]["metrics"]["nll"]["value"] = 0.515626
        assert not check_artifact(beyond, _baseline([rule])).passed

    def test_rtol_and_combined_behavior(self) -> None:
        rule = _nll_rule(atol=0.0, rtol=0.02)  # allows +/- 0.01 at expected 0.5
        ok = _artifact()
        ok["sections"]["metrics"]["nll"]["value"] = 0.5099
        assert check_artifact(ok, _baseline([rule])).passed
        combined = _nll_rule(atol=0.001, rtol=0.02)  # 0.011 total
        edge = _artifact()
        edge["sections"]["metrics"]["nll"]["value"] = 0.5109
        assert check_artifact(edge, _baseline([combined])).passed

    def test_no_worse_both_directions_and_boundary(self) -> None:
        higher = CheckRule(
            path="/sections/metrics/accuracy",
            comparator=ComparatorType.NO_WORSE,
            expected=0.75,
            provenance="simulated",
            direction="higher_is_better",
            degradation_atol=0.01,
            rationale="test",
        )
        improved = _artifact()
        improved["sections"]["metrics"]["accuracy"]["value"] = 0.80
        assert check_artifact(improved, _baseline([higher])).passed
        boundary = _artifact()
        boundary["sections"]["metrics"]["accuracy"]["value"] = 0.74
        assert check_artifact(boundary, _baseline([higher])).passed  # inclusive
        worse = _artifact()
        worse["sections"]["metrics"]["accuracy"]["value"] = 0.7399
        assert not check_artifact(worse, _baseline([higher])).passed
        lower = CheckRule(
            path="/sections/metrics/nll",
            comparator=ComparatorType.NO_WORSE,
            expected=0.5,
            provenance="simulated",
            direction="lower_is_better",
            degradation_atol=0.0,
        )
        better_nll = _artifact()
        better_nll["sections"]["metrics"]["nll"]["value"] = 0.4
        assert check_artifact(better_nll, _baseline([lower])).passed
        worse_nll = _artifact()
        worse_nll["sections"]["metrics"]["nll"]["value"] = 0.51
        assert not check_artifact(worse_nll, _baseline([lower])).passed

    def test_structure_keys_and_extras(self) -> None:
        rule = CheckRule(
            path="/sections/metrics",
            comparator=ComparatorType.STRUCTURE,
            required_keys=["nll", "count", "accuracy"],
        )
        assert check_artifact(_artifact(), _baseline([rule])).passed
        extra = _artifact()
        extra["sections"]["metrics"]["surprise"] = {"value": 1, "provenance": "measured"}
        report = check_artifact(extra, _baseline([rule]))
        assert not report.passed  # extras rejected by default
        allowing = CheckRule(
            path="/sections/metrics",
            comparator=ComparatorType.STRUCTURE,
            required_keys=["nll", "count", "accuracy"],
            allow_extra_keys=True,
        )
        assert check_artifact(extra, _baseline([allowing])).passed

    def test_missing_required_field_is_regression(self) -> None:
        gone = _artifact()
        del gone["sections"]["metrics"]["nll"]
        report = check_artifact(gone, _baseline([_nll_rule()]))
        assert report.exit_code == 1
        assert "missing" in report.failures[0].explanation

    def test_provenance_change_fails_with_same_value(self) -> None:
        relabeled = _artifact()
        relabeled["sections"]["metrics"]["nll"]["provenance"] = "measured"
        report = check_artifact(relabeled, _baseline([_nll_rule()]))
        assert report.exit_code == 1
        assert report.failures[0].comparator == "provenance"

    def test_nan_rejected(self) -> None:
        bad = _artifact()
        bad["sections"]["metrics"]["nll"]["value"] = float("nan")
        report = check_artifact(bad, _baseline([_nll_rule()]))
        assert not report.passed
        assert "non-finite" in report.failures[0].explanation


class TestClassification:
    def test_incompatible_type_and_schema_are_harness_errors(self) -> None:
        wrong_type = _artifact(artifact_type="other")
        with pytest.raises(HarnessError, match="incompatible artifact type"):
            check_artifact(wrong_type, _baseline([_nll_rule()]))
        wrong_schema = _artifact(artifact_schema_version=9)
        with pytest.raises(HarnessError, match="artifact_schema_version"):
            check_artifact(wrong_schema, _baseline([_nll_rule()]))

    def test_environment_gate_is_harness_error(self) -> None:
        wrong_torch = _artifact()
        wrong_torch["environment"]["torch"] = "2.13.0"
        with pytest.raises(HarnessError, match="environment gate"):
            check_artifact(wrong_torch, _baseline([_nll_rule()]))

    def test_python_allowed_set(self) -> None:
        py312 = _artifact()
        py312["environment"]["python_minor"] = "3.12"
        assert check_artifact(py312, _baseline([_nll_rule()])).passed
        py310 = _artifact()
        py310["environment"]["python_minor"] = "3.10"
        with pytest.raises(HarnessError, match="allowed set"):
            check_artifact(py310, _baseline([_nll_rule()]))


class TestDiffDeterminism:
    def test_paths_sorted_and_output_deterministic(self, tmp_path: Path) -> None:
        rules = [
            _nll_rule(),
            CheckRule(
                path="/sections/metrics/count",
                comparator=ComparatorType.EXACT,
                expected=64,
                provenance="measured",
            ),
            CheckRule(path="/sections/name", comparator=ComparatorType.EXACT, expected="fixture"),
        ]
        baseline = _baseline(rules)
        report = check_artifact(_artifact(), baseline)
        paths = [e.path for e in report.entries]
        assert paths == sorted(paths)
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        write_diff(check_artifact(_artifact(), baseline), a)
        write_diff(check_artifact(_artifact(), baseline), b)
        assert a.read_bytes() == b.read_bytes()
        payload = diff_payload(report, failure_category="none")
        assert "timestamp" not in json.dumps(payload)

    def test_diff_has_no_absolute_paths(self, tmp_path: Path) -> None:
        report = check_artifact(_artifact(), _baseline([_nll_rule()]))
        out = tmp_path / "diff.json"
        write_diff(report, out)
        assert str(tmp_path) not in out.read_text()


class TestCapture:
    def test_overwrite_protection_and_flag(self, tmp_path: Path) -> None:
        artifact = generate_smoke_artifact()
        out = tmp_path / "smoke.json"
        capture_baseline(artifact, out)
        with pytest.raises(HarnessError, match="code-review"):
            capture_baseline(artifact, out)
        _, comparison = capture_baseline(artifact, out, overwrite=True)
        assert comparison is not None
        assert comparison["changed_expectations"] == {}
        assert comparison["added_paths"] == [] and comparison["removed_paths"] == []

    def test_capture_rejects_foreign_artifact(self, tmp_path: Path) -> None:
        with pytest.raises(HarnessError, match="artifact_type"):
            capture_baseline(_artifact(), tmp_path / "x.json")


class TestSmokeRoundTrip:
    def test_generate_check_and_deliberate_perturbation(self, tmp_path: Path) -> None:
        artifact = generate_smoke_artifact()
        baseline = build_smoke_baseline(artifact, capture_command="test")
        report = check_artifact(artifact, baseline)
        assert report.passed, [e.to_payload() for e in report.failures]

        # Deliberate perturbation of one realistic numerical field: the
        # diff must identify the exact path, values, tolerances, and
        # regression category.
        perturbed = copy.deepcopy(artifact)
        perturbed["sections"]["w4a4"]["nll"]["value"] += 0.01
        failing = check_artifact(perturbed, baseline)
        assert failing.exit_code == 1
        assert len(failing.failures) == 1
        entry = failing.failures[0]
        assert entry.path == "/sections/w4a4/nll"
        assert entry.expected == artifact["sections"]["w4a4"]["nll"]["value"]
        assert entry.actual == pytest.approx(entry.expected + 0.01)
        assert entry.atol == 1e-6 and entry.rtol == 1e-5
        out = tmp_path / "diff.json"
        write_diff(failing, out)
        payload = json.loads(out.read_text())
        assert payload["verdict"] == "fail"
        assert payload["failure_category"] == "regression"

    def test_committed_baseline_matches_fresh_artifact(self) -> None:
        committed = load_baseline("tests/baselines/smoke.json")
        report = check_artifact(generate_smoke_artifact(), committed)
        assert report.passed, [e.to_payload() for e in report.failures]
