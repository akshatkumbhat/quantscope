"""Accounting determinism/reconciliation and cost-invariant tests
(ADR-014; internal consistency, not realism)."""

import math
from typing import ClassVar

import pytest
import torch

from quantscope.benchmark import benchmark_config
from quantscope.hardware import (
    GROUP_ORDER_V1,
    GroupAccount,
    account_model,
    config_identifier,
    configuration_cost,
    group_cost,
    load_hardware_profile,
    recommend_for_budget,
)
from quantscope.models.tiny_cnn import build_model

PROFILE = load_hardware_profile("configs/hardware/generic_edge_npu.yaml").profile


def _model():
    torch.manual_seed(0)
    return build_model(benchmark_config(seed=0).model)


def _accounting():
    return account_model(_model())


ALL8 = [(8, 8)] * len(GROUP_ORDER_V1)
ALL4 = [(4, 4)] * len(GROUP_ORDER_V1)


class TestAccounting:
    def test_deterministic_digest(self) -> None:
        assert _accounting().digest() == _accounting().digest()

    def test_group_order_is_canonical(self) -> None:
        accounting = _accounting()
        assert tuple(g.name for g in accounting.groups) == GROUP_ORDER_V1
        assert accounting.group_order_version == "group-order-v1"

    def test_totals_reconcile_with_groups(self) -> None:
        accounting = _accounting()
        assert accounting.total_parameters == sum(g.parameters for g in accounting.groups)
        assert accounting.total_macs == sum(g.macs for g in accounting.groups)
        model_params = sum(
            m.weight.numel()
            for m in _model().modules()
            if isinstance(m, torch.nn.Conv2d | torch.nn.Linear)
        )
        assert accounting.total_parameters == model_params

    def test_tensor_identity_reconciliation(self) -> None:
        # Reconcile against actual traced tensors, not aggregate counts:
        # every modeled tensor must match the numel of the real ReLU
        # output (batch excluded), and the modeled set must be exactly
        # {input} + ReLU sites.
        model = _model()
        actual: dict[str, int] = {}
        handles = [
            module.register_forward_hook(
                lambda _m, _i, out, *, _name=name: actual.__setitem__(_name, out[0].numel())
            )
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.ReLU)
        ]
        with torch.no_grad():
            model(torch.zeros(1, 1, 32, 32))
        for h in handles:
            h.remove()

        accounting = _accounting()
        modeled = {t.site: t for g in accounting.groups for t in g.tensors}
        assert set(modeled) == set(actual) | {"__input__"}
        for site, numel in actual.items():
            assert modeled[site].elements == numel, site
            assert modeled[site].traffic == "read+write"
        assert modeled["__input__"].traffic == "read"
        assert modeled["__input__"].producer_group == "stem"
        assert modeled["__input__"].elements == 32 * 32

    def test_exclusions_recorded(self) -> None:
        excluded = _accounting().excluded_operations
        for kind in (
            "batchnorm2d",
            "residual_add",
            "adaptive_avg_pool2d",
            "flatten",
            "output_logits_tensor",
            "quantize_dequantize_boundaries",
        ):
            assert excluded[kind] >= 1, kind
        assert excluded["residual_add"] == 2


class TestCostInvariants:
    def test_all_int4_cheaper_than_all_int8(self) -> None:
        accounting = _accounting()
        assert (
            configuration_cost(accounting, ALL4, PROFILE).total
            < configuration_cost(accounting, ALL8, PROFILE).total
        )

    def test_total_equals_component_sum_exactly(self) -> None:
        cost = configuration_cost(_accounting(), ALL4, PROFILE)
        components = cost.components
        assert cost.total == (
            components.compute
            + components.weight_memory
            + components.activation_memory
            + components.overhead
        )
        assert math.isclose(
            cost.total, sum(c.total for c in cost.per_group.values()), rel_tol=1e-12
        )

    def test_components_finite_nonnegative(self) -> None:
        for assignment in (ALL8, ALL4):
            for comp in configuration_cost(_accounting(), assignment, PROFILE).per_group.values():
                for value in (
                    comp.compute,
                    comp.weight_memory,
                    comp.activation_memory,
                    comp.overhead,
                ):
                    assert math.isfinite(value) and value >= 0.0

    def test_group_monotonicity_component_and_whole(self) -> None:
        accounting = _accounting()
        base = configuration_cost(accounting, ALL8, PROFILE)
        for i, name in enumerate(GROUP_ORDER_V1):
            assignment = list(ALL8)
            assignment[i] = (4, 4)
            lowered = configuration_cost(accounting, assignment, PROFILE)
            b, low = base.per_group[name], lowered.per_group[name]
            assert low.compute <= b.compute
            assert low.weight_memory < b.weight_memory
            if accounting.group(name).tensors:
                assert low.activation_memory < b.activation_memory
            assert lowered.total < base.total  # whole-configuration level

    def test_changing_one_group_touches_only_that_group(self) -> None:
        accounting = _accounting()
        base = configuration_cost(accounting, ALL8, PROFILE)
        assignment = list(ALL8)
        assignment[0] = (4, 4)
        changed = configuration_cost(accounting, assignment, PROFILE)
        for name in GROUP_ORDER_V1[1:]:
            assert changed.per_group[name] == base.per_group[name]

    def test_zero_layer_group_costs_zero(self) -> None:
        empty = GroupAccount(name="empty", layers=(), tensors=())
        comp = group_cost(empty, 4, 4, PROFILE)
        assert comp.total == 0.0

    def test_unsupported_pair_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="W2A2"):
            configuration_cost(_accounting(), [(2, 2)] * len(GROUP_ORDER_V1), PROFILE)

    def test_identifier_canonical_and_readable(self) -> None:
        ident = config_identifier([*ALL4[:7], (8, 8)])
        assert ident.startswith("stem=w4a4|block_a_conv1=w4a4|")
        assert ident.endswith("classifier=w8a8")
        with pytest.raises(ValueError, match="group assignments"):
            config_identifier(ALL4[:3])


class TestRecommendations:
    RECORDS: ClassVar[list[dict]] = [
        {
            "identifier": "a",
            "bits": [4] * 8,
            "nll": 0.20,
            "accuracy": 0.90,
            "normalized_cost": 0.55,
        },
        {
            "identifier": "b",
            "bits": [8] * 8,
            "nll": 0.10,
            "accuracy": 0.95,
            "normalized_cost": 1.00,
        },
        {
            "identifier": "c",
            "bits": [4] * 8,
            "nll": 0.15,
            "accuracy": 0.93,
            "normalized_cost": 0.70,
        },
        # exact-NLL tie with c at lower cost:
        {
            "identifier": "d",
            "bits": [4] * 8,
            "nll": 0.15,
            "accuracy": 0.92,
            "normalized_cost": 0.65,
        },
    ]

    def test_lowest_nll_within_budget(self) -> None:
        result = recommend_for_budget(self.RECORDS, 0.75)
        assert result["feasible"] is True
        assert result["recommendation"]["identifier"] == "d"  # tie -> lower cost

    def test_tie_breaks_accuracy_then_identifier(self) -> None:
        records = [
            {"identifier": "z", "bits": [], "nll": 0.15, "accuracy": 0.93, "normalized_cost": 0.65},
            {"identifier": "y", "bits": [], "nll": 0.15, "accuracy": 0.93, "normalized_cost": 0.65},
            {"identifier": "x", "bits": [], "nll": 0.15, "accuracy": 0.94, "normalized_cost": 0.65},
        ]
        result = recommend_for_budget(records, 0.75)
        assert result["recommendation"]["identifier"] == "x"  # higher accuracy first
        result = recommend_for_budget(records[:2], 0.75)
        assert result["recommendation"]["identifier"] == "y"  # lexicographic last resort

    def test_infeasible_budget_structured_not_relaxed(self) -> None:
        result = recommend_for_budget(self.RECORDS, 0.50)
        assert result == {
            "budget": 0.50,
            "feasible": False,
            "cheapest_available_normalized_cost": 0.55,
            "cheapest_configuration": "a",
            "recommendation": None,
        }
