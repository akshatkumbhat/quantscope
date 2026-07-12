"""Uniform simulated quantization of a torch model (simulation policy v1).

Every number produced through this path is **simulated**: fake-quantized
FP32 arithmetic, not integer-kernel execution.

Simulation policy v1 (documented so results are interpretable):

- **Weights**: every ``Conv2d``/``Linear`` weight is fake-quantized
  per-output-channel (axis 0), symmetric, signed, at ``weight_bits``.
  Biases stay FP32 (matching common backend practice).
- **Activations**: the model input and the output of every ``ReLU`` module
  are fake-quantized per-tensor, asymmetric, at ``act_bits`` (unsigned for
  post-ReLU tensors, signed for the input). Ranges are calibrated with
  ``MinMaxObserver`` over a calibration dataset.
- BatchNorm is *not* folded and logits are not quantized. This does not
  match backend INT8 semantics exactly; the backend-matched profile is a
  separate, later deliverable (plan step C).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from quantscope.observers import MinMaxObserver
from quantscope.quantization.affine import (
    Granularity,
    QuantParams,
    Scheme,
    compute_quant_params,
    fake_quantize,
)

__all__ = ["SimQuantConfig", "simulate_quantized"]

logger = logging.getLogger(__name__)

_WEIGHT_MODULES = (nn.Conv2d, nn.Linear)
_INPUT_KEY = "__input__"


@dataclass(frozen=True)
class SimQuantConfig:
    """Uniform simulated quantization setting (e.g. W8A8, W4A4, W4A8)."""

    weight_bits: int
    act_bits: int

    @property
    def label(self) -> str:
        return f"W{self.weight_bits}A{self.act_bits}"


def _fake_quantize_weights(model: nn.Module, bits: int) -> int:
    """Fake-quantize all Conv2d/Linear weights in place. Returns layer count."""
    count = 0
    for module in model.modules():
        if isinstance(module, _WEIGHT_MODULES):
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
        raise ValueError("model contains no Conv2d/Linear layers to quantize")
    return count


def _activation_sites(model: nn.Module) -> dict[str, nn.Module]:
    """Named ReLU modules whose outputs get quantized under policy v1."""
    sites = {name: module for name, module in model.named_modules() if isinstance(module, nn.ReLU)}
    if not sites:
        raise ValueError("model contains no ReLU modules; policy v1 cannot apply")
    return sites


def _calibrate_activations(
    model: nn.Module,
    calibration: Dataset,
    *,
    bits: int,
    batch_size: int,
) -> dict[str, QuantParams]:
    """Observe input + ReLU-output ranges over the calibration set."""
    sites = _activation_sites(model)
    observers: dict[str, MinMaxObserver] = {
        name: MinMaxObserver(bits=bits, signed=False, scheme=Scheme.ASYMMETRIC) for name in sites
    }
    observers[_INPUT_KEY] = MinMaxObserver(bits=bits, signed=True, scheme=Scheme.ASYMMETRIC)

    handles = []
    for name, module in sites.items():

        def hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor, *, _name: str = name) -> None:
            observers[_name].observe(out.detach().numpy())

        handles.append(module.register_forward_hook(hook))

    loader = DataLoader(calibration, batch_size=batch_size, shuffle=False)
    seen = 0
    with torch.no_grad():
        for images, _ in loader:
            observers[_INPUT_KEY].observe(images.numpy())
            model(images)
            seen += 1
    for handle in handles:
        handle.remove()
    if seen == 0:
        raise ValueError("calibration dataset produced no batches")
    return {name: obs.to_quant_params() for name, obs in observers.items()}


def simulate_quantized(
    model: nn.Module,
    calibration: Dataset,
    config: SimQuantConfig,
    *,
    batch_size: int = 64,
) -> nn.Module:
    """Return a deep-copied model with simulated uniform quantization applied.

    The original model is untouched. The returned model runs FP32 arithmetic
    with fake-quantized weights and activation clamping — a **simulation**,
    not integer execution.
    """
    model = copy.deepcopy(model).eval()
    n_layers = _fake_quantize_weights(model, config.weight_bits)
    act_params = _calibrate_activations(
        model, calibration, bits=config.act_bits, batch_size=batch_size
    )

    def _fq(tensor: torch.Tensor, params: QuantParams) -> torch.Tensor:
        return torch.from_numpy(fake_quantize(tensor.detach().numpy().astype(np.float32), params))

    for name, module in _activation_sites(model).items():
        params = act_params[name]

        def hook(
            _mod: nn.Module,
            _inp: tuple,
            out: torch.Tensor,
            *,
            _params: QuantParams = params,
        ) -> torch.Tensor:
            return _fq(out, _params)

        module.register_forward_hook(hook)

    input_params = act_params[_INPUT_KEY]
    model.register_forward_pre_hook(lambda _mod, inp: (_fq(inp[0], input_params), *inp[1:]))
    logger.info(
        "simulated %s: %d weight layers, %d activation sites",
        config.label,
        n_layers,
        len(act_params) - 1,
    )
    return model
