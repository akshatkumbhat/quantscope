"""Impulse stress-design gate evaluation (ADR-012 addenda 2 and 4).

Pure decision logic for the stress-design gates, separated from the
data/model drivers in ``scripts/`` so the criteria are unit-testable.
The preregistered Gate v3 constants live here (``GATE_V3_SPEC``,
``GATE_V3_STRESS``) and are imported by the runner; changing them after
the gate has produced an artifact is a protocol violation, not a tweak.

Gate v3 (ADR-012 addendum 4) reuses the Gate v2 criteria verbatim with
one design change: impulse magnitude 6-sigma -> 7-sigma. The 2.0x
input-expansion threshold is retained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "GATE_V3_SPEC",
    "GATE_V3_STRESS",
    "ImpulseStressSpec",
    "StressGateResult",
    "StressGateSpec",
    "evaluate_stress_gate",
]


@dataclass(frozen=True)
class ImpulseStressSpec:
    """Preregistered impulse-stress intervention parameters."""

    fraction: float
    magnitude: float
    seed: int


@dataclass(frozen=True)
class StressGateSpec:
    """Preregistered pass/fail criteria for an impulse stress-design gate.

    ``early_sites`` is the structural site set fixed in ADR-012
    addendum 2: the input observer plus the activation observers before
    the first spatial-downsampling boundary (``down_conv``, stride 2).
    """

    early_sites: tuple[str, ...]
    input_site: str = "__input__"
    min_early_expanded: int = 3
    site_expansion_threshold: float = 1.25
    input_expansion_threshold: float = 2.0
    nll_degradation_threshold: float = 0.02


@dataclass(frozen=True)
class StressGateResult:
    """Outcome of one gate evaluation. ``failures`` is empty iff passed."""

    passed: bool
    early_expanded: int
    input_ratio: float
    nll_degradation: float
    pairing_ok: bool
    failures: tuple[str, ...] = field(default_factory=tuple)


def evaluate_stress_gate(
    spec: StressGateSpec,
    scale_ratios: Mapping[str, float],
    nll_degradation: float,
    pairing_ok: bool,
) -> StressGateResult:
    """Apply the preregistered criteria to measured gate quantities.

    ``scale_ratios`` maps observer site name to stressed/clean MinMax
    scale ratio; it must cover every eligible early site and the input
    site (a missing site is an evaluation bug, not a gate failure).
    """
    missing = [s for s in (*spec.early_sites, spec.input_site) if s not in scale_ratios]
    if missing:
        raise KeyError(f"scale_ratios missing required sites: {missing}")

    early_expanded = sum(
        scale_ratios[site] >= spec.site_expansion_threshold for site in spec.early_sites
    )
    input_ratio = float(scale_ratios[spec.input_site])

    failures: list[str] = []
    if early_expanded < spec.min_early_expanded:
        failures.append(
            f"1: early-site reach {early_expanded}/{len(spec.early_sites)}"
            f" < {spec.min_early_expanded}"
        )
    if input_ratio < spec.input_expansion_threshold:
        failures.append(
            f"1: input expansion {input_ratio:.2f}x < {spec.input_expansion_threshold}x"
        )
    if nll_degradation <= spec.nll_degradation_threshold:
        failures.append(
            f"2: degradation {nll_degradation:+.4f} <= {spec.nll_degradation_threshold}"
        )
    if not pairing_ok:
        failures.append("3: pairing integrity violated")

    return StressGateResult(
        passed=not failures,
        early_expanded=early_expanded,
        input_ratio=input_ratio,
        nll_degradation=float(nll_degradation),
        pairing_ok=pairing_ok,
        failures=tuple(failures),
    )


# Gate v3 preregistration (ADR-012 addendum 4). Sole design change from
# Gate v2: magnitude 6.0 -> 7.0. Criteria, thresholds, fraction, and the
# structural early-site set are unchanged.
GATE_V3_STRESS = ImpulseStressSpec(fraction=0.002, magnitude=7.0, seed=1006)

GATE_V3_SPEC = StressGateSpec(
    early_sites=("__input__", "stem_relu", "block_a.relu1", "block_a.relu_out"),
)
