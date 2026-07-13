"""Exhaustive INT4/INT8 mixed-precision sweep over quantization groups.

With 8 groups and two choices each there are 256 configurations — small
enough to evaluate every one, giving *exact* optima and Pareto frontiers
to judge search heuristics against (ADR-010).

Provenance: NLL/accuracy are **simulated** (policy v1 fake-quant); the
weight-bits cost is **estimated** (normalized weight-storage bits; the
analytical hardware cost model is a later deliverable).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from torch import nn
from torch.utils.data import Dataset

from quantscope.evaluation.loop import evaluate_detailed
from quantscope.quantization.simulate import (
    BOTTLENECK_RESNET_GROUPS,
    GroupSpec,
    SimQuantConfig,
    simulate_quantized_groups,
)

__all__ = [
    "SweepRecord",
    "config_cost",
    "exhaustive_sweep",
    "group_param_counts",
    "pareto_frontier",
]

logger = logging.getLogger(__name__)

_BIT_CHOICES = (4, 8)


@dataclass(frozen=True)
class SweepRecord:
    """One evaluated mixed-precision configuration."""

    bits: tuple[int, ...]  # per group, in group-dict order
    cost: float  # estimated normalized weight-bits cost
    nll: float  # simulated
    accuracy: float  # simulated

    def to_dict(self) -> dict[str, object]:
        return {
            "bits": list(self.bits),
            "cost": self.cost,
            "nll": self.nll,
            "accuracy": self.accuracy,
        }


def group_param_counts(
    model: nn.Module, groups: Mapping[str, GroupSpec] = BOTTLENECK_RESNET_GROUPS
) -> dict[str, int]:
    """Weight-parameter count per quantization group."""
    modules = dict(model.named_modules())
    counts: dict[str, int] = {}
    for name, spec in groups.items():
        counts[name] = sum(modules[w].weight.numel() for w in spec.weights)
    return counts


def config_cost(bits: tuple[int, ...], param_counts: dict[str, int]) -> float:
    """Normalized weight-storage cost: all-INT8 = 1.0, all-INT4 = 0.5. Estimated."""
    names = list(param_counts)
    if len(bits) != len(names):
        raise ValueError(f"expected {len(names)} bit choices, got {len(bits)}")
    total = sum(param_counts[n] * b for n, b in zip(names, bits, strict=True))
    return total / (sum(param_counts.values()) * 8)


def exhaustive_sweep(
    model: nn.Module,
    calibration: Dataset,
    test_set: Dataset,
    *,
    groups: Mapping[str, GroupSpec] = BOTTLENECK_RESNET_GROUPS,
    batch_size: int = 64,
) -> list[SweepRecord]:
    """Evaluate every INT4/INT8 group assignment. Slow (2^n_groups evals)."""
    names = list(groups)
    param_counts = group_param_counts(model, groups)
    records: list[SweepRecord] = []
    total = 2 ** len(names)
    for i, bits in enumerate(itertools.product(_BIT_CHOICES, repeat=len(names))):
        assignment = {name: SimQuantConfig(b, b) for name, b in zip(names, bits, strict=True)}
        quantized = simulate_quantized_groups(
            model, calibration, assignment, groups=groups, batch_size=batch_size
        )
        detailed = evaluate_detailed(quantized, test_set)
        records.append(
            SweepRecord(
                bits=bits,
                cost=config_cost(bits, param_counts),
                nll=detailed["nll"],
                accuracy=detailed["accuracy"],
            )
        )
        if (i + 1) % 32 == 0:
            logger.info("sweep progress: %d/%d", i + 1, total)
    return records


def pareto_frontier(records: list[SweepRecord], *, quality: str = "nll") -> list[SweepRecord]:
    """Configs not dominated in (cost, quality); lower is better for both
    when quality='nll', higher-accuracy-lower-cost when quality='accuracy'."""
    if quality not in ("nll", "accuracy"):
        raise ValueError(f"unknown quality key: {quality!r}")

    def better(v_new: float, v_old: float) -> bool:
        return v_new < v_old if quality == "nll" else v_new > v_old

    # Sort so that at equal cost the BEST quality comes first; a later
    # equal-cost candidate can then never wrongly join the frontier.
    sign = 1.0 if quality == "nll" else -1.0
    frontier: list[SweepRecord] = []
    for candidate in sorted(records, key=lambda r: (r.cost, sign * getattr(r, quality))):
        q = getattr(candidate, quality)
        if not frontier:
            frontier.append(candidate)
            continue
        best_so_far = getattr(frontier[-1], quality)
        if better(q, best_so_far):
            frontier.append(candidate)
    return frontier
