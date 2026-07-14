"""Simulated quantization of a torch model (simulation policy v1).

Every number produced through this path is **simulated**: fake-quantized
FP32 arithmetic, not integer-kernel execution.

Simulation policy v1 (documented so results are interpretable):

- **Weights**: ``Conv2d``/``Linear`` weights are fake-quantized
  per-output-channel (axis 0), symmetric, signed. Biases stay FP32
  (matching common backend practice).
- **Activations**: the model input and the outputs of ``ReLU`` modules are
  fake-quantized per-tensor, asymmetric (unsigned for post-ReLU tensors,
  signed for the input), min-max calibrated over a calibration dataset.
- BatchNorm is *not* folded and logits are not quantized. This does not
  match backend INT8 semantics exactly; the backend-matched profile is a
  separate, later deliverable (plan step C).

Two entry points:

- :func:`simulate_quantized` — uniform bit widths over the whole model.
- :func:`simulate_quantized_groups` — per-group bit widths (or FP32
  passthrough) over a named :class:`GroupSpec` partition; used by the
  sensitivity ablation and mixed-precision search.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from quantscope.observers import CalibrationObserver, MinMaxObserver
from quantscope.quantization.affine import (
    Granularity,
    QuantParams,
    Scheme,
    compute_quant_params,
    fake_quantize,
)

__all__ = [
    "BOTTLENECK_RESNET_GROUPS",
    "GroupSpec",
    "SimQuantConfig",
    "calibrate_activation_params",
    "quantize_weights_uniform",
    "simulate_quantized",
    "simulate_quantized_groups",
    "simulate_quantized_with_params",
]

logger = logging.getLogger(__name__)

_WEIGHT_MODULES = (nn.Conv2d, nn.Linear)
_INPUT_KEY = "__input__"


@dataclass(frozen=True)
class SimQuantConfig:
    """Simulated quantization setting for a model or group (e.g. W4A4)."""

    weight_bits: int
    act_bits: int

    @property
    def label(self) -> str:
        return f"W{self.weight_bits}A{self.act_bits}"


@dataclass(frozen=True)
class GroupSpec:
    """One quantization group: named weight modules + named ReLU sites.

    ``include_input`` marks the group owning model-input quantization.
    """

    weights: tuple[str, ...]
    activations: tuple[str, ...]
    include_input: bool = False


# Eight-group partition of models.bottleneck_resnet.BottleneckResNet
# (ADR-008 / plan step B). Two INT4/INT8 choices per group => 256 total
# mixed-precision configurations, small enough to search exhaustively.
BOTTLENECK_RESNET_GROUPS: dict[str, GroupSpec] = {
    "stem": GroupSpec(("stem_conv",), ("stem_relu",), include_input=True),
    "block_a_conv1": GroupSpec(("block_a.conv1",), ("block_a.relu1",)),
    "block_a_conv2": GroupSpec(("block_a.conv2",), ("block_a.relu_out",)),
    "down": GroupSpec(("down_conv",), ("down_relu",)),
    "block_b": GroupSpec(("block_b.conv1", "block_b.conv2"), ("block_b.relu1", "block_b.relu_out")),
    "bottleneck": GroupSpec(("bottleneck_conv",), ("bottleneck_relu",)),
    "expand": GroupSpec(("expand_conv",), ("expand_relu",)),
    # Classifier logits stay FP32 under policy v1: weight-only group.
    "classifier": GroupSpec(("classifier",), ()),
}


def _named_module(model: nn.Module, name: str) -> nn.Module:
    module = dict(model.named_modules()).get(name)
    if module is None:
        raise ValueError(f"model has no module named {name!r}")
    return module


def _fake_quantize_weights(model: nn.Module, bits_by_module: Mapping[str, int]) -> int:
    """Fake-quantize the named Conv2d/Linear weights in place."""
    count = 0
    for name, bits in bits_by_module.items():
        module = _named_module(model, name)
        if not isinstance(module, _WEIGHT_MODULES):
            raise ValueError(f"{name!r} is not a Conv2d/Linear module")
        weight = module.weight.detach().numpy()
        params = compute_quant_params(
            weight,
            bits=bits,
            signed=True,
            scheme=Scheme.SYMMETRIC,
            granularity=Granularity.PER_CHANNEL,
            channel_axis=0,
        )
        module.weight.data = torch.from_numpy(fake_quantize(weight, params))
        count += 1
    if count == 0:
        raise ValueError("no weights selected for quantization")
    return count


def _all_relu_sites(model: nn.Module) -> dict[str, nn.Module]:
    sites = {name: module for name, module in model.named_modules() if isinstance(module, nn.ReLU)}
    if not sites:
        raise ValueError("model contains no ReLU modules; policy v1 cannot apply")
    return sites


def _calibrate_activations(
    model: nn.Module,
    calibration: Dataset,
    *,
    bits_by_site: Mapping[str, int],
    input_bits: int | None,
    batch_size: int,
    observer_factory: type[CalibrationObserver] = MinMaxObserver,
) -> dict[str, QuantParams]:
    """Observe the selected ReLU outputs (and optionally the input).

    ``observer_factory`` selects the activation calibration policy
    (ADR-012); weights are never affected by it.
    """
    observers: dict[str, CalibrationObserver] = {
        name: observer_factory(bits=bits, signed=False, scheme=Scheme.ASYMMETRIC)
        for name, bits in bits_by_site.items()
    }
    if input_bits is not None:
        observers[_INPUT_KEY] = observer_factory(
            bits=input_bits, signed=True, scheme=Scheme.ASYMMETRIC
        )

    handles = []
    for name in bits_by_site:
        module = _named_module(model, name)

        def hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor, *, _name: str = name) -> None:
            observers[_name].observe(out.detach().numpy())

        handles.append(module.register_forward_hook(hook))

    loader = DataLoader(calibration, batch_size=batch_size, shuffle=False)
    seen = 0
    with torch.no_grad():
        for images, _ in loader:
            if input_bits is not None:
                observers[_INPUT_KEY].observe(images.numpy())
            model(images)
            seen += 1
    for handle in handles:
        handle.remove()
    if seen == 0:
        raise ValueError("calibration dataset produced no batches")
    return {name: obs.to_quant_params() for name, obs in observers.items()}


def _attach_fake_quant(model: nn.Module, act_params: dict[str, QuantParams]) -> None:
    def _fq(tensor: torch.Tensor, params: QuantParams) -> torch.Tensor:
        return torch.from_numpy(fake_quantize(tensor.detach().numpy().astype(np.float32), params))

    for name, params in act_params.items():
        if name == _INPUT_KEY:
            continue
        module = _named_module(model, name)

        def hook(
            _mod: nn.Module,
            _inp: tuple,
            out: torch.Tensor,
            *,
            _params: QuantParams = params,
        ) -> torch.Tensor:
            return _fq(out, _params)

        module.register_forward_hook(hook)

    if _INPUT_KEY in act_params:
        input_params = act_params[_INPUT_KEY]
        model.register_forward_pre_hook(lambda _mod, inp: (_fq(inp[0], input_params), *inp[1:]))


def calibrate_activation_params(
    model: nn.Module,
    calibration: Dataset,
    *,
    bits: int,
    batch_size: int = 64,
    observer_factory: type[CalibrationObserver] = MinMaxObserver,
) -> dict[str, QuantParams]:
    """Public policy-v1 activation calibration (input + all ReLU sites).

    Used by the ADR-012 observer study to inspect per-site scales and
    ranges without building a full simulated model.
    """
    sites = _all_relu_sites(model)
    return _calibrate_activations(
        model,
        calibration,
        bits_by_site=dict.fromkeys(sites, bits),
        input_bits=bits,
        batch_size=batch_size,
        observer_factory=observer_factory,
    )


def simulate_quantized(
    model: nn.Module,
    calibration: Dataset,
    config: SimQuantConfig,
    *,
    batch_size: int = 64,
    observer_factory: type[CalibrationObserver] = MinMaxObserver,
) -> nn.Module:
    """Uniform simulated quantization of the whole model (policy v1).

    The original model is untouched. The returned deep copy runs FP32
    arithmetic with fake-quantized weights and activations — a
    **simulation**, not integer execution. ``observer_factory`` varies
    the activation calibration policy only (ADR-012); weights always use
    per-channel symmetric min-max.
    """
    model = copy.deepcopy(model).eval()
    weight_names = [name for name, m in model.named_modules() if isinstance(m, _WEIGHT_MODULES)]
    n_layers = _fake_quantize_weights(model, dict.fromkeys(weight_names, config.weight_bits))
    sites = _all_relu_sites(model)
    act_params = _calibrate_activations(
        model,
        calibration,
        bits_by_site=dict.fromkeys(sites, config.act_bits),
        input_bits=config.act_bits,
        batch_size=batch_size,
        observer_factory=observer_factory,
    )
    _attach_fake_quant(model, act_params)
    logger.info(
        "simulated %s: %d weight layers, %d activation sites",
        config.label,
        n_layers,
        len(act_params) - 1,
    )
    return model


def quantize_weights_uniform(model: nn.Module, *, bits: int) -> nn.Module:
    """Deep copy of ``model`` with every Conv2d/Linear weight fake-quantized
    (per-channel symmetric, policy v1); activations untouched.

    Policy-v1 activation calibration observes the weight-quantized model
    (:func:`simulate_quantized` does this internally). This public step
    lets callers calibrate several observer policies against one
    weight-quantized copy and still match :func:`simulate_quantized`
    exactly when the parameters are re-attached via
    :func:`simulate_quantized_with_params`.
    """
    model = copy.deepcopy(model).eval()
    weight_names = [name for name, m in model.named_modules() if isinstance(m, _WEIGHT_MODULES)]
    _fake_quantize_weights(model, dict.fromkeys(weight_names, bits))
    return model


def simulate_quantized_with_params(
    model: nn.Module,
    act_params: Mapping[str, QuantParams],
    *,
    weight_bits: int,
) -> nn.Module:
    """Policy-v1 simulation from **precomputed** activation parameters.

    Same weight treatment as :func:`simulate_quantized` (uniform
    per-channel symmetric min-max at ``weight_bits``), but activation
    fake-quantization uses the caller's ``act_params`` instead of
    calibrating. This is the ADR-012 mechanism-decomposition entry
    point: it lets clean- and stressed-calibration parameters be mixed
    site by site. ``act_params`` must cover the input key and every
    ReLU site so a silently unquantized site is impossible.
    """
    model = copy.deepcopy(model).eval()
    required = set(_all_relu_sites(model)) | {_INPUT_KEY}
    if set(act_params) != required:
        missing = required - set(act_params)
        extra = set(act_params) - required
        raise ValueError(
            f"act_params must cover exactly the policy-v1 sites; missing={sorted(missing)}"
            f" extra={sorted(extra)}"
        )
    weight_names = [name for name, m in model.named_modules() if isinstance(m, _WEIGHT_MODULES)]
    _fake_quantize_weights(model, dict.fromkeys(weight_names, weight_bits))
    _attach_fake_quant(model, dict(act_params))
    return model


def simulate_quantized_groups(
    model: nn.Module,
    calibration: Dataset,
    assignment: Mapping[str, SimQuantConfig | None],
    *,
    groups: Mapping[str, GroupSpec] = BOTTLENECK_RESNET_GROUPS,
    batch_size: int = 64,
) -> nn.Module:
    """Per-group simulated quantization; ``None`` leaves a group in FP32.

    ``assignment`` must cover every key in ``groups`` explicitly so a
    forgotten group is an error, not a silent FP32 passthrough.
    """
    if set(assignment) != set(groups):
        missing = set(groups) ^ set(assignment)
        raise ValueError(f"assignment must cover exactly the group names; mismatch: {missing}")

    model = copy.deepcopy(model).eval()
    weight_bits: dict[str, int] = {}
    site_bits: dict[str, int] = {}
    input_bits: int | None = None
    for group_name, cfg in assignment.items():
        if cfg is None:
            continue
        spec = groups[group_name]
        for w in spec.weights:
            weight_bits[w] = cfg.weight_bits
        for a in spec.activations:
            site_bits[a] = cfg.act_bits
        if spec.include_input:
            input_bits = cfg.act_bits

    if not weight_bits and not site_bits and input_bits is None:
        return model  # all groups FP32: a valid no-op copy

    if weight_bits:
        _fake_quantize_weights(model, weight_bits)
    act_params = _calibrate_activations(
        model,
        calibration,
        bits_by_site=site_bits,
        input_bits=input_bits,
        batch_size=batch_size,
    )
    _attach_fake_quant(model, act_params)
    active = {g: c.label for g, c in assignment.items() if c is not None}
    logger.info("simulated groups %s", active)
    return model
