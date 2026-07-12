"""Numerical-error metrics for comparing reference vs. quantized tensors.

Backend-independent (numpy-only, ADR-002). These metrics quantify what
quantization did to a tensor or layer output; they make no claim about
hardware behavior. Metric provenance: values computed here are *measured*
properties of the two arrays supplied (which may themselves come from
simulated quantization — the caller labels the run, see ADR-004).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "ErrorMetrics",
    "compare",
    "cosine_similarity",
    "max_abs_error",
    "mse",
    "saturation_rate",
    "sqnr_db",
]


@dataclass(frozen=True)
class ErrorMetrics:
    """Error metrics between a reference tensor and its approximation.

    ``sqnr_db`` and ``cosine`` are ``inf``/``1.0`` respectively for a
    perfect match, and ``sqnr_db`` is ``-inf`` when the reference has zero
    power but the approximation does not (documented, not silent).
    """

    mse: float
    sqnr_db: float
    cosine: float
    max_abs_error: float
    num_elements: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialize for JSON/CSV artifact output."""
        return asdict(self)


def _validate_pair(
    reference: np.ndarray, approximation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    approximation = np.asarray(approximation, dtype=np.float64)
    if reference.shape != approximation.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} vs approximation {approximation.shape}"
        )
    if reference.size == 0:
        raise ValueError("cannot compute metrics on empty tensors")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(approximation)):
        raise ValueError("inputs contain NaN or infinite values")
    return reference, approximation


def mse(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Mean squared error."""
    reference, approximation = _validate_pair(reference, approximation)
    return float(np.mean((reference - approximation) ** 2))


def max_abs_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Maximum absolute elementwise error."""
    reference, approximation = _validate_pair(reference, approximation)
    return float(np.max(np.abs(reference - approximation)))


def sqnr_db(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in decibels.

    ``10 * log10(signal_power / noise_power)``.

    Degenerate cases (documented, not silent):
        - zero noise -> ``inf`` (perfect reconstruction)
        - zero signal power and nonzero noise -> ``-inf``
        - zero signal power and zero noise -> ``inf``
    """
    reference, approximation = _validate_pair(reference, approximation)
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean((reference - approximation) ** 2))
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def cosine_similarity(reference: np.ndarray, approximation: np.ndarray) -> float:
    """Cosine similarity between flattened tensors.

    Degenerate cases: if both tensors are zero vectors the similarity is
    defined as ``1.0`` (identical); if exactly one is zero it is ``0.0``.
    """
    reference, approximation = _validate_pair(reference, approximation)
    a = reference.ravel()
    b = approximation.ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def saturation_rate(quantized: np.ndarray, qmin: int, qmax: int) -> float:
    """Fraction of quantized values pinned at either integer range bound.

    A high rate indicates the calibration range clipped real signal.

    Raises:
        ValueError: for empty input or ``qmin >= qmax``.
    """
    quantized = np.asarray(quantized)
    if quantized.size == 0:
        raise ValueError("cannot compute saturation rate on an empty tensor")
    if qmin >= qmax:
        raise ValueError(f"qmin ({qmin}) must be < qmax ({qmax})")
    saturated = np.count_nonzero(quantized <= qmin) + np.count_nonzero(quantized >= qmax)
    return float(saturated / quantized.size)


def compare(reference: np.ndarray, approximation: np.ndarray) -> ErrorMetrics:
    """Compute the full error-metric set between two tensors."""
    reference, approximation = _validate_pair(reference, approximation)
    return ErrorMetrics(
        mse=mse(reference, approximation),
        sqnr_db=sqnr_db(reference, approximation),
        cosine=cosine_similarity(reference, approximation),
        max_abs_error=max_abs_error(reference, approximation),
        num_elements=int(reference.size),
    )
