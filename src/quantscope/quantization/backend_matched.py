"""Graph-anchored backend-matched simulator (ADR-011, plan step C).

Purpose, narrowly defined: hold Torch's fusion, placement, calibration
statistics, and graph topology constant while replacing Torch's affine
quantize/dequantize arithmetic with QuantScope's arithmetic. The strict
``sim_backend_matched ↔ reference_fx`` comparison then isolates
arithmetic semantics — scale/zero-point computation, rounding, clipping —
from graph plumbing.

Version-pinned to Torch 2.2.x: the histogram-bounds extractor touches
``HistogramObserver._non_linear_param_search()`` (private). That
dependency is contained HERE and nowhere else in QuantScope, guarded by
a version assertion and a characterization test.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.ao.quantization.observer import (
    HistogramObserver,
    PerChannelMinMaxObserver,
)

from quantscope.quantization.affine import (
    Granularity,
    QuantParams,
    Scheme,
    compute_quant_params,
    fake_quantize,
)

__all__ = [
    "FrozenQuantScopeFakeQuant",
    "ObserverBounds",
    "build_graph_anchored_sim",
    "extract_histogram_bounds",
    "quantscope_params_from_bounds",
    "weighted_modules",
]

logger = logging.getLogger(__name__)

_SUPPORTED_TORCH = "2.2"


def _assert_torch_version() -> None:
    if not torch.__version__.startswith(_SUPPORTED_TORCH):
        raise RuntimeError(
            f"graph-anchored simulator is version-pinned to torch {_SUPPORTED_TORCH}.x "
            f"(found {torch.__version__}); re-validate the private observer "
            "internals before lifting this guard (ADR-011)"
        )


@dataclass(frozen=True)
class ObserverBounds:
    """An observer's selected calibration range plus its configuration."""

    low: np.ndarray
    high: np.ndarray
    quant_min: int
    quant_max: int
    signed: bool
    scheme: Scheme
    granularity: Granularity
    channel_axis: int | None
    raw_low: np.ndarray | None = None  # pre-search histogram extent, if any
    raw_high: np.ndarray | None = None


def extract_histogram_bounds(observer: HistogramObserver) -> ObserverBounds:
    """Extract a HistogramObserver's searched clipping range.

    Contained private-API access (`_non_linear_param_search`); Torch's
    public ``calculate_qparams()`` calls the same method for its bounds.
    Records both the raw histogram extent and the searched extent.
    """
    _assert_torch_version()
    if not isinstance(observer, HistogramObserver):
        raise TypeError(f"expected HistogramObserver, got {type(observer).__name__}")
    searched_min, searched_max = observer._non_linear_param_search()
    signed = observer.dtype != torch.quint8
    return ObserverBounds(
        low=searched_min.detach().numpy().reshape(()),
        high=searched_max.detach().numpy().reshape(()),
        quant_min=int(observer.quant_min),
        quant_max=int(observer.quant_max),
        signed=signed,
        scheme=Scheme.ASYMMETRIC,  # torch.per_tensor_affine
        granularity=Granularity.PER_TENSOR,
        channel_axis=None,
        raw_low=observer.min_val.detach().numpy().reshape(()),
        raw_high=observer.max_val.detach().numpy().reshape(()),
    )


def _bits_from_range(quant_min: int, quant_max: int, *, signed: bool) -> int:
    """Map a torch quant range to a QuantScope bit width.

    fbgemm's reduce_range activations use [0, 127] == 7-bit unsigned;
    weights use [-128, 127] == 8-bit signed. Anything else is rejected
    rather than approximated.
    """
    levels = quant_max - quant_min + 1
    bits = int(np.log2(levels))
    if 2**bits != levels:
        raise ValueError(f"unsupported quant range [{quant_min}, {quant_max}]")
    if signed and quant_min != -(2 ** (bits - 1)):
        raise ValueError(f"unsupported signed range [{quant_min}, {quant_max}]")
    if not signed and quant_min != 0:
        raise ValueError(f"unsupported unsigned range [{quant_min}, {quant_max}]")
    return bits


def quantscope_params_from_bounds(
    bounds: ObserverBounds, *, qparam_policy: str = "torch_2_2"
) -> QuantParams:
    """QuantScope qparams from an observer's frozen bounds (ADR-011)."""
    bits = _bits_from_range(bounds.quant_min, bounds.quant_max, signed=bounds.signed)
    if bounds.granularity is Granularity.PER_CHANNEL:
        values = np.stack([bounds.low, bounds.high]).astype(np.float32)
        params = compute_quant_params(
            values,
            bits=bits,
            signed=bounds.signed,
            scheme=bounds.scheme,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=-1,
            qparam_policy=qparam_policy,
        )
        return QuantParams(
            scale=params.scale,
            zero_point=params.zero_point,
            qmin=params.qmin,
            qmax=params.qmax,
            bits=params.bits,
            signed=params.signed,
            scheme=params.scheme,
            granularity=params.granularity,
            channel_axis=bounds.channel_axis,
        )
    values = np.array([bounds.low, bounds.high], dtype=np.float32)
    return compute_quant_params(
        values,
        bits=bits,
        signed=bounds.signed,
        scheme=bounds.scheme,
        granularity=Granularity.PER_TENSOR,
        qparam_policy=qparam_policy,
    )


class FrozenQuantScopeFakeQuant(nn.Module):
    """Applies QuantScope fake-quant with frozen qparams; never updates stats."""

    def __init__(self, params: QuantParams) -> None:
        super().__init__()
        self.params = params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fake_quantize(x.detach().numpy().astype(np.float32), self.params)
        return torch.from_numpy(out).to(dtype=x.dtype, device=x.device)

    def extra_repr(self) -> str:
        return (
            f"scale~{np.asarray(self.params.scale).ravel()[0]:.6g}, "
            f"zp~{np.asarray(self.params.zero_point).ravel()[0]}, "
            f"q[{self.params.qmin},{self.params.qmax}]"
        )


def _module_weight(module: nn.Module) -> nn.Parameter:
    """Weight of a plain or fused (intrinsic Sequential) weighted module."""
    if hasattr(module, "weight") and module.weight is not None:
        return module.weight
    if isinstance(module, nn.Sequential) and hasattr(module[0], "weight"):
        return module[0].weight
    raise ValueError(f"no weight found on {type(module).__name__}")


def weighted_modules(prepared: nn.Module) -> dict[str, nn.Module]:
    """Weighted modules that conversion would quantize (qconfig attached)."""
    out: dict[str, nn.Module] = {}
    for name, module in prepared.named_modules():
        if getattr(module, "qconfig", None) is None:
            continue
        try:
            _module_weight(module)
        except ValueError:
            continue
        # Skip children of an already-collected fused module.
        if any(name.startswith(parent + ".") for parent in out):
            continue
        out[name] = module
    return out


def extract_weight_bounds(module: nn.Module) -> ObserverBounds:
    """Run the module's configured weight observer on its (folded) weight,
    matching conversion behavior, and freeze the resulting ranges."""
    _assert_torch_version()
    observer = module.qconfig.weight()
    if not isinstance(observer, PerChannelMinMaxObserver):
        raise TypeError(f"expected PerChannelMinMaxObserver, got {type(observer).__name__}")
    weight = _module_weight(module)
    observer(weight.detach())
    return ObserverBounds(
        low=observer.min_val.detach().numpy(),
        high=observer.max_val.detach().numpy(),
        quant_min=int(observer.quant_min),
        quant_max=int(observer.quant_max),
        signed=observer.dtype != torch.quint8,
        scheme=Scheme.SYMMETRIC,  # torch.per_channel_symmetric
        granularity=Granularity.PER_CHANNEL,
        channel_axis=int(observer.ch_axis),
    )


def build_graph_anchored_sim(
    prepared: nn.Module,
    *,
    quantize_activations: bool = True,
    quantize_weights: bool = True,
    qparam_policy: str = "torch_2_2",
) -> tuple[nn.Module, dict[str, QuantParams]]:
    """Build the graph-anchored backend-matched simulator.

    Deep-copies the *calibrated* prepared-FX model; swaps every activation
    observer for a :class:`FrozenQuantScopeFakeQuant` (or ``nn.Identity``
    when ``quantize_activations`` is False); fake-quantizes each weighted
    module's folded weight via the configured torch weight observer's
    ranges + QuantScope arithmetic (bias stays FP32; fused structure
    preserved). Returns the model and the qparams per site for artifacts.
    """
    _assert_torch_version()
    model = copy.deepcopy(prepared).eval()
    all_params: dict[str, QuantParams] = {}

    for name, module in list(model.named_modules()):
        if isinstance(module, HistogramObserver):
            if quantize_activations:
                bounds = extract_histogram_bounds(module)
                params = quantscope_params_from_bounds(bounds, qparam_policy=qparam_policy)
                all_params[name] = params
                replacement: nn.Module = FrozenQuantScopeFakeQuant(params)
            else:
                replacement = nn.Identity()
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, replacement)

    if quantize_weights:
        for name, module in weighted_modules(model).items():
            bounds = extract_weight_bounds(module)
            params = quantscope_params_from_bounds(bounds, qparam_policy=qparam_policy)
            all_params[f"{name}.weight"] = params
            weight = _module_weight(module)
            weight.data = torch.from_numpy(fake_quantize(weight.detach().numpy(), params))

    logger.info(
        "graph-anchored sim built: %d activation sites, %d weight sites (policy=%s)",
        sum(1 for k in all_params if not k.endswith(".weight")),
        sum(1 for k in all_params if k.endswith(".weight")),
        qparam_policy,
    )
    return model, all_params
