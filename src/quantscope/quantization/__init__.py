"""Quantization arithmetic and framework adapters."""

from quantscope.quantization.affine import (
    MAX_BITS,
    MIN_BITS,
    Granularity,
    IntegerRange,
    QuantParams,
    Scheme,
    compute_quant_params,
    dequantize,
    fake_quantize,
    integer_range,
    power_of_two_scale,
    quantize,
)

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
