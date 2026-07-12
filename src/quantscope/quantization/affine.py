"""Backend-independent affine quantization arithmetic.

Implements the affine mapping ``real ≈ scale * (q - zero_point)`` with
symmetric/asymmetric schemes, per-tensor/per-channel granularity,
configurable bit widths, and power-of-two scale approximation.

This module deliberately depends only on NumPy (ADR-002): it is the
numerical core reused by observers, PTQ/QAT adapters, and simulation.
All quantization here is *simulated* arithmetic — it makes no claim about
real integer-kernel execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

__all__ = [
    "MAX_BITS",
    "MIN_BITS",
    "Granularity",
    "IntegerRange",
    "QuantParams",
    "Scheme",
    "compute_quant_params",
    "dequantize",
    "fake_quantize",
    "integer_range",
    "power_of_two_scale",
    "quantize",
]

# Bit widths outside this window are either meaningless (< 2 cannot
# represent a sign or more than saturation) or beyond any edge-accelerator
# integer path this project models (> 16).
MIN_BITS = 2
MAX_BITS = 16

# Smallest representable range half-width before a tensor is treated as
# constant; guards scale > 0 without silently absorbing real signal.
_EPS = float(np.finfo(np.float32).eps)


class Scheme(StrEnum):
    """Quantization scheme."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class Granularity(StrEnum):
    """Quantization granularity."""

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"


@dataclass(frozen=True)
class IntegerRange:
    """Inclusive integer range representable at a given bit width."""

    qmin: int
    qmax: int
    bits: int
    signed: bool
    narrow_range: bool


@dataclass(frozen=True)
class QuantParams:
    """Quantization parameters plus the metadata needed to reproduce them.

    ``scale`` and ``zero_point`` are scalars for per-tensor granularity and
    1-D arrays (one entry per channel) for per-channel granularity.
    """

    scale: np.ndarray
    zero_point: np.ndarray
    qmin: int
    qmax: int
    bits: int
    signed: bool
    scheme: Scheme
    granularity: Granularity
    channel_axis: int | None = None

    def __post_init__(self) -> None:
        scale = np.asarray(self.scale, dtype=np.float32)
        zero_point = np.asarray(self.zero_point, dtype=np.int32)
        if scale.shape != zero_point.shape:
            raise ValueError(f"scale shape {scale.shape} != zero_point shape {zero_point.shape}")
        if self.granularity is Granularity.PER_CHANNEL:
            if self.channel_axis is None:
                raise ValueError("per-channel params require channel_axis")
            if scale.ndim != 1:
                raise ValueError("per-channel scale must be 1-D (one entry per channel)")
        elif scale.ndim != 0:
            raise ValueError("per-tensor scale must be scalar")
        if not np.all(scale > 0):
            raise ValueError("scale must be strictly positive")
        if np.any(zero_point < self.qmin) or np.any(zero_point > self.qmax):
            raise ValueError("zero_point outside [qmin, qmax]")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "zero_point", zero_point)


def integer_range(bits: int, *, signed: bool, narrow_range: bool = False) -> IntegerRange:
    """Return the inclusive integer range for a bit width.

    ``narrow_range`` drops the most negative signed value (e.g. -128 for
    INT8) so the range is symmetric around zero; commonly used for weights.

    Raises:
        ValueError: if ``bits`` is outside [MIN_BITS, MAX_BITS] or
            ``narrow_range`` is requested for an unsigned range.
    """
    if not isinstance(bits, int) or isinstance(bits, bool):
        raise ValueError(f"bits must be an int, got {type(bits).__name__}")
    if bits < MIN_BITS or bits > MAX_BITS:
        raise ValueError(f"bits must be in [{MIN_BITS}, {MAX_BITS}], got {bits}")
    if signed:
        qmin = -(2 ** (bits - 1)) + (1 if narrow_range else 0)
        qmax = 2 ** (bits - 1) - 1
    else:
        if narrow_range:
            raise ValueError("narrow_range is only defined for signed ranges")
        qmin = 0
        qmax = 2**bits - 1
    return IntegerRange(qmin=qmin, qmax=qmax, bits=bits, signed=signed, narrow_range=narrow_range)


def _validate_finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite values")


def _minmax_params(
    low: np.ndarray,
    high: np.ndarray,
    int_range: IntegerRange,
    scheme: Scheme,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (scale, zero_point) arrays from per-slice min/max arrays.

    A constant or all-zero slice yields ``scale = 1.0, zero_point = 0``
    (any scale reconstructs a constant exactly through the zero point),
    documented rather than silent: the range is genuinely empty.
    """
    qmin, qmax = int_range.qmin, int_range.qmax
    if scheme is Scheme.SYMMETRIC:
        bound = np.maximum(np.abs(low), np.abs(high)).astype(np.float64)
        # Zero point: 0 for signed, mid-range for unsigned symmetric.
        zp = 0 if int_range.signed else (qmax + qmin + 1) // 2
        # Scale divides by the positive half-range (qmax - zp), matching the
        # common convention (e.g. PyTorch): both ±bound stay representable;
        # the most negative code may go unused rather than saturating +bound.
        half_range = float(qmax - zp)
        degenerate = bound <= _EPS
        scale = np.where(degenerate, 1.0, bound / half_range)
        zero_point = np.full_like(scale, zp, dtype=np.int32)
    else:
        # Asymmetric: range must include zero so that zero is exactly
        # representable (required for zero padding to be lossless).
        low = np.minimum(low, 0.0).astype(np.float64)
        high = np.maximum(high, 0.0).astype(np.float64)
        span = high - low
        degenerate = span <= _EPS
        scale = np.where(degenerate, 1.0, span / (qmax - qmin))
        zero_point = np.where(
            degenerate,
            0,
            np.clip(np.round(qmin - low / scale), qmin, qmax),
        ).astype(np.int32)
    return scale.astype(np.float32), zero_point


def compute_quant_params(
    values: np.ndarray,
    *,
    bits: int = 8,
    signed: bool = True,
    scheme: Scheme = Scheme.ASYMMETRIC,
    granularity: Granularity = Granularity.PER_TENSOR,
    channel_axis: int | None = None,
    narrow_range: bool = False,
) -> QuantParams:
    """Compute affine quantization parameters from observed values.

    Args:
        values: observed tensor (calibration data or weights).
        bits: integer bit width in [MIN_BITS, MAX_BITS].
        signed: signed or unsigned integer range.
        scheme: symmetric or asymmetric.
        granularity: per-tensor or per-channel.
        channel_axis: required axis for per-channel granularity.
        narrow_range: drop the most negative signed value.

    Raises:
        ValueError: for empty input, NaN/infinite input, invalid bit width,
            or an invalid/missing channel axis.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("cannot compute quantization parameters from an empty tensor")
    _validate_finite(values, "values")
    int_range = integer_range(bits, signed=signed, narrow_range=narrow_range)

    if granularity is Granularity.PER_TENSOR:
        if channel_axis is not None:
            raise ValueError("channel_axis is only valid for per-channel granularity")
        low = np.asarray(values.min())
        high = np.asarray(values.max())
        scale, zero_point = _minmax_params(low, high, int_range, scheme)
        scale, zero_point = scale.reshape(()), zero_point.reshape(())
    else:
        if channel_axis is None:
            raise ValueError("per-channel granularity requires channel_axis")
        if not -values.ndim <= channel_axis < values.ndim:
            raise ValueError(f"channel_axis {channel_axis} out of range for ndim {values.ndim}")
        channel_axis = channel_axis % values.ndim
        reduce_axes = tuple(ax for ax in range(values.ndim) if ax != channel_axis)
        low = values.min(axis=reduce_axes)
        high = values.max(axis=reduce_axes)
        scale, zero_point = _minmax_params(low, high, int_range, scheme)

    return QuantParams(
        scale=scale,
        zero_point=zero_point,
        qmin=int_range.qmin,
        qmax=int_range.qmax,
        bits=bits,
        signed=signed,
        scheme=scheme,
        granularity=granularity,
        channel_axis=channel_axis if granularity is Granularity.PER_CHANNEL else None,
    )


def _broadcast_params(params: QuantParams, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """Reshape per-channel scale/zero-point for broadcasting against data."""
    if params.granularity is Granularity.PER_TENSOR:
        return params.scale, params.zero_point
    assert params.channel_axis is not None
    if not -ndim <= params.channel_axis < ndim:
        raise ValueError(f"channel_axis {params.channel_axis} out of range for ndim {ndim}")
    shape = [1] * ndim
    shape[params.channel_axis % ndim] = params.scale.shape[0]
    return params.scale.reshape(shape), params.zero_point.reshape(shape)


def quantize(values: np.ndarray, params: QuantParams) -> np.ndarray:
    """Quantize real values to integers: ``q = clip(round(x/scale) + zp)``.

    Uses round-half-to-even (NumPy's ``round``), saturating at
    ``[qmin, qmax]``. Returns int32 (wide enough for every supported bit
    width; the logical width is ``params.bits``).

    Raises:
        ValueError: if ``values`` contains NaN/infinity, or the channel
            dimension does not match the per-channel parameters.
    """
    values = np.asarray(values, dtype=np.float32)
    _validate_finite(values, "values")
    scale, zero_point = _broadcast_params(params, values.ndim)
    if params.granularity is Granularity.PER_CHANNEL:
        axis = params.channel_axis % values.ndim  # type: ignore[operator]
        if values.shape[axis] != params.scale.shape[0]:
            raise ValueError(
                f"values has {values.shape[axis]} channels on axis {axis}, "
                f"params expect {params.scale.shape[0]}"
            )
    q = np.round(values.astype(np.float64) / scale) + zero_point
    return np.clip(q, params.qmin, params.qmax).astype(np.int32)


def dequantize(quantized: np.ndarray, params: QuantParams) -> np.ndarray:
    """Reconstruct real values: ``x_hat = scale * (q - zero_point)``.

    Raises:
        ValueError: if ``quantized`` lies outside ``[qmin, qmax]``.
    """
    quantized = np.asarray(quantized)
    if np.any(quantized < params.qmin) or np.any(quantized > params.qmax):
        raise ValueError(f"quantized values outside [{params.qmin}, {params.qmax}]")
    scale, zero_point = _broadcast_params(params, quantized.ndim)
    return (scale.astype(np.float64) * (quantized - zero_point)).astype(np.float32)


def fake_quantize(values: np.ndarray, params: QuantParams) -> np.ndarray:
    """Quantize then dequantize (simulated quantization error in FP32)."""
    return dequantize(quantize(values, params), params)


def power_of_two_scale(scale: float | np.ndarray) -> np.ndarray:
    """Approximate positive scales by the nearest power of two.

    Rounding policy: round ``log2(scale)`` half-up to the nearest integer
    exponent, i.e. ``2 ** round(log2(scale))``; exact midpoints in log space
    round toward the larger exponent. This keeps the approximation within a
    factor of ``sqrt(2)`` of the input and yields shift-only rescaling on
    hardware without multipliers.

    Raises:
        ValueError: if any scale is non-positive, NaN, or infinite.
    """
    arr = np.asarray(scale, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("scale must be finite and strictly positive")
    exponents = np.floor(np.log2(arr) + 0.5)
    return np.power(2.0, exponents).astype(np.float32)
