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

__all__ = [
    "ErrorMetrics",
    "compare",
    "cosine_similarity",
    "max_abs_error",
    "mse",
    "saturation_rate",
    "sqnr_db",
]
