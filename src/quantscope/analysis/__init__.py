"""Numerical analysis: error metrics, layer capture, regression triage."""

from quantscope.analysis.metrics import (
    ErrorMetrics,
    compare,
    cosine_similarity,
    max_abs_error,
    mse,
    saturation_rate,
    sqnr_db,
)
from quantscope.analysis.stress_gate import (
    GATE_V3_SPEC,
    GATE_V3_STRESS,
    ImpulseStressSpec,
    StressGateResult,
    StressGateSpec,
    evaluate_stress_gate,
)

__all__ = [
    "GATE_V3_SPEC",
    "GATE_V3_STRESS",
    "ErrorMetrics",
    "ImpulseStressSpec",
    "StressGateResult",
    "StressGateSpec",
    "compare",
    "cosine_similarity",
    "evaluate_stress_gate",
    "max_abs_error",
    "mse",
    "saturation_rate",
    "sqnr_db",
]
