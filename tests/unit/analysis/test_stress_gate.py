"""Unit tests for the stress-design gate logic (ADR-012 addenda 2 and 4)."""

import pytest

from quantscope.analysis.stress_gate import (
    GATE_V3_SPEC,
    GATE_V3_STRESS,
    evaluate_stress_gate,
)

_PASSING_RATIOS = {
    "__input__": 2.3,
    "stem_relu": 1.9,
    "block_a.relu1": 1.5,
    "block_a.relu_out": 1.30,
    "down_relu": 1.05,
    "block_b.relu1": 1.0,
}


class TestGateV3Preregistration:
    """The frozen ADR-012 addendum 4 constants. A failure here means the
    preregistered design was edited after the fact."""

    def test_sole_design_change_is_magnitude(self) -> None:
        assert GATE_V3_STRESS.magnitude == 7.0  # 6.0 in Gate v2
        assert GATE_V3_STRESS.fraction == 0.002  # unchanged
        assert GATE_V3_STRESS.seed == 1006  # 1000 + dev seed convention

    def test_thresholds_retained_from_v2(self) -> None:
        assert GATE_V3_SPEC.input_expansion_threshold == 2.0
        assert GATE_V3_SPEC.site_expansion_threshold == 1.25
        assert GATE_V3_SPEC.min_early_expanded == 3
        assert GATE_V3_SPEC.nll_degradation_threshold == 0.02

    def test_structural_early_site_set_unchanged(self) -> None:
        assert GATE_V3_SPEC.early_sites == (
            "__input__",
            "stem_relu",
            "block_a.relu1",
            "block_a.relu_out",
        )
        assert GATE_V3_SPEC.input_site == "__input__"


class TestEvaluateStressGate:
    def test_pass_when_all_criteria_met(self) -> None:
        result = evaluate_stress_gate(GATE_V3_SPEC, _PASSING_RATIOS, 0.11, True)
        assert result.passed
        assert result.failures == ()
        assert result.early_expanded == 4
        assert result.input_ratio == pytest.approx(2.3)

    def test_fail_on_early_site_reach(self) -> None:
        ratios = {**_PASSING_RATIOS, "block_a.relu1": 1.1, "block_a.relu_out": 1.2}
        result = evaluate_stress_gate(GATE_V3_SPEC, ratios, 0.11, True)
        assert not result.passed
        assert result.early_expanded == 2
        assert any(f.startswith("1: early-site reach 2/4") for f in result.failures)

    def test_fail_on_input_expansion_gate_v2_scenario(self) -> None:
        # The exact Gate v2 miss: 1.96x vs the retained 2.0x threshold.
        ratios = {**_PASSING_RATIOS, "__input__": 1.96}
        result = evaluate_stress_gate(GATE_V3_SPEC, ratios, 0.11, True)
        assert not result.passed
        assert result.failures == ("1: input expansion 1.96x < 2.0x",)

    def test_input_threshold_is_inclusive(self) -> None:
        ratios = {**_PASSING_RATIOS, "__input__": 2.0}
        assert evaluate_stress_gate(GATE_V3_SPEC, ratios, 0.11, True).passed

    def test_fail_on_degradation_boundary(self) -> None:
        # The behavioral criterion is strictly greater-than 0.02.
        result = evaluate_stress_gate(GATE_V3_SPEC, _PASSING_RATIOS, 0.02, True)
        assert not result.passed
        assert any(f.startswith("2: degradation") for f in result.failures)

    def test_fail_on_pairing_violation(self) -> None:
        result = evaluate_stress_gate(GATE_V3_SPEC, _PASSING_RATIOS, 0.11, False)
        assert not result.passed
        assert "3: pairing integrity violated" in result.failures

    def test_multiple_failures_all_reported(self) -> None:
        ratios = dict.fromkeys(_PASSING_RATIOS, 1.0)
        result = evaluate_stress_gate(GATE_V3_SPEC, ratios, 0.0, False)
        assert not result.passed
        assert len(result.failures) == 4

    def test_missing_required_site_is_an_error_not_a_failure(self) -> None:
        ratios = {k: v for k, v in _PASSING_RATIOS.items() if k != "stem_relu"}
        with pytest.raises(KeyError, match="stem_relu"):
            evaluate_stress_gate(GATE_V3_SPEC, ratios, 0.11, True)
