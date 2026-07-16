"""Component-wise analytical cost calculation (ADR-014).

All arithmetic is float64 and no component is rounded before totals,
normalization, Pareto construction, feasibility checks, or tie-breaks;
rounding is display-only. Every output is an **estimate** of the
modeled quantizable workload — never whole-model energy, latency, or
total hardware cost, and never a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quantscope.hardware.accounting import GROUP_ORDER_V1, GroupAccount, ModelAccounting
from quantscope.hardware.profile import HardwareProfile

__all__ = [
    "ComponentCost",
    "ConfigurationCost",
    "config_identifier",
    "configuration_cost",
    "group_cost",
    "recommend_for_budget",
]


@dataclass(frozen=True)
class ComponentCost:
    """Estimated cost components for one group (ncu, float64)."""

    compute: float
    weight_memory: float
    activation_memory: float
    overhead: float

    @property
    def total(self) -> float:
        return self.compute + self.weight_memory + self.activation_memory + self.overhead

    def to_dict(self) -> dict[str, float]:
        return {
            "compute_ncu": self.compute,
            "weight_memory_ncu": self.weight_memory,
            "activation_memory_ncu": self.activation_memory,
            "overhead_ncu": self.overhead,
            "total_ncu": self.total,
        }


@dataclass(frozen=True)
class ConfigurationCost:
    """Estimated cost of one full mixed-precision assignment."""

    identifier: str
    assignment: tuple[tuple[int, int], ...]  # (w, a) per group, canonical order
    per_group: dict[str, ComponentCost]

    @property
    def components(self) -> ComponentCost:
        return ComponentCost(
            compute=sum(c.compute for c in self.per_group.values()),
            weight_memory=sum(c.weight_memory for c in self.per_group.values()),
            activation_memory=sum(c.activation_memory for c in self.per_group.values()),
            overhead=sum(c.overhead for c in self.per_group.values()),
        )

    @property
    def total(self) -> float:
        return self.components.total


def config_identifier(assignment: Sequence[tuple[int, int]]) -> str:
    """Readable canonical identifier, e.g. ``stem=w4a4|block_a_conv1=w8a8|...``.

    Group order is the versioned GROUP_ORDER_V1 constant, never
    incidental dict order. Used for the final lexicographic tie-break.
    """
    if len(assignment) != len(GROUP_ORDER_V1):
        raise ValueError(f"expected {len(GROUP_ORDER_V1)} group assignments, got {len(assignment)}")
    return "|".join(
        f"{name}=w{w}a{a}" for name, (w, a) in zip(GROUP_ORDER_V1, assignment, strict=True)
    )


def group_cost(
    group: GroupAccount, weight_bits: int, activation_bits: int, profile: HardwareProfile
) -> ComponentCost:
    """Component-wise estimated cost of one group at one precision pair."""
    coefficient = profile.compute_coefficient(weight_bits, activation_bits)
    compute = float(group.macs) * coefficient
    weight_memory = float(group.parameters) * float(weight_bits) * profile.weight_memory_ncu_per_bit
    activation_elements = 0.0
    for tensor in group.tensors:
        # Declared traffic model: produced tensors move twice (one read
        # + one write); the model input moves once (read), owned by the
        # stem group. No per-consumer multiplier.
        multiplier = 1.0 if tensor.traffic == "read" else 2.0
        activation_elements += multiplier * float(tensor.elements)
    activation_memory = (
        activation_elements * float(activation_bits) * profile.activation_memory_ncu_per_bit
    )
    overhead = float(len(group.layers)) * profile.per_layer_overhead_ncu
    return ComponentCost(
        compute=compute,
        weight_memory=weight_memory,
        activation_memory=activation_memory,
        overhead=overhead,
    )


def configuration_cost(
    accounting: ModelAccounting,
    assignment: Sequence[tuple[int, int]],
    profile: HardwareProfile,
) -> ConfigurationCost:
    """Estimated cost of a full assignment ((w, a) per group, canonical order)."""
    if len(assignment) != len(accounting.groups):
        raise ValueError(
            f"assignment covers {len(assignment)} groups; accounting has {len(accounting.groups)}"
        )
    per_group = {
        group.name: group_cost(group, w, a, profile)
        for group, (w, a) in zip(accounting.groups, assignment, strict=True)
    }
    return ConfigurationCost(
        identifier=config_identifier(assignment),
        assignment=tuple(assignment),
        per_group=per_group,
    )


def recommend_for_budget(
    records: Sequence[Mapping],
    budget: float,
) -> dict:
    """Budget recommendation over enriched records (ADR-014 semantics).

    Each record needs: ``normalized_cost``, ``nll``, ``accuracy``,
    ``identifier``. Recommends the lowest simulated NLL with
    normalized estimated cost <= budget; exact-NLL ties break by
    (a) lower cost, (b) higher accuracy, (c) lexicographic identifier.
    Infeasible budgets return a structured result and are never
    relaxed.
    """
    feasible = [r for r in records if r["normalized_cost"] <= budget]
    if not feasible:
        cheapest = min(records, key=lambda r: (r["normalized_cost"], r["identifier"]))
        return {
            "budget": budget,
            "feasible": False,
            "cheapest_available_normalized_cost": cheapest["normalized_cost"],
            "cheapest_configuration": cheapest["identifier"],
            "recommendation": None,
        }
    best = min(
        feasible,
        key=lambda r: (r["nll"], r["normalized_cost"], -r["accuracy"], r["identifier"]),
    )
    return {
        "budget": budget,
        "feasible": True,
        "recommendation": {
            "identifier": best["identifier"],
            "bits": best["bits"],
            "nll": best["nll"],
            "accuracy": best["accuracy"],
            "normalized_cost": best["normalized_cost"],
        },
    }
