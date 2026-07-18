"""Torch-native differentiable fake quantization for ADR-013 QAT.

This module is a **differentiable adapter**, not a replacement core:
the NumPy affine core (`quantization.affine`) remains the
backend-independent reference implementation, and the adapter must
match its forward quantize/dequantize behavior exactly (unit-tested
parity on deterministic tensors). No NumPy is called from the
differentiable forward path.

Fixed-quantization-specification QAT (ADR-013 + addendum):

- **Activation qparams are numerically frozen** — computed once by the
  frozen PTQ calibration procedure and never updated during training.
- **Weight scales are recomputed each forward** from the *current*
  weights under the frozen per-channel symmetric min-max rule, and are
  detached from autograd (no gradient flows through scale
  computation).
- **Clipped STE** (declared gradient policy, identical for weights and
  activations): d(fake_quant)/dx = 1 where the pre-clamp integer code
  lies in [qmin, qmax], 0 where the value saturates.
- Fake quantization is active from the first fine-tuning step through
  the last; there is no warm-up, delay, observer update, staged bit
  width, or batch-dependent recalibration.

Everything simulated here is fake-quant arithmetic in FP32/FP64 — not
integer execution.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset

from quantscope.quantization.affine import _EPS, Granularity, QuantParams, integer_range
from quantscope.quantization.simulate import _INPUT_KEY, _WEIGHT_MODULES, _all_relu_sites

__all__ = [
    "QATRecipe",
    "fp32_finetune",
    "nll_gap_recovery",
    "qat_finetune",
    "torch_fake_quantize",
]


def nll_gap_recovery(ptq_nll: float, qat_nll: float, fp32_nll: float) -> float:
    """ADR-013 primary metric: fraction of the PTQ-to-FP32 NLL gap
    recovered by QAT, ``(PTQ - QAT) / (PTQ - FP32)``, reported
    UNCLIPPED (values > 1 or < 0 are returned as computed)."""
    gap = ptq_nll - fp32_nll
    if gap == 0.0:
        raise ZeroDivisionError("PTQ and FP32 NLL identical: recovery fraction undefined")
    return (ptq_nll - qat_nll) / gap


logger = logging.getLogger(__name__)


class _FakeQuantSTE(torch.autograd.Function):
    """Fake quantize with the clipped straight-through estimator.

    Forward matches the NumPy reference bit-for-bit: the division and
    rounding happen in float64 against the float32-valued scale
    (upcast), round-half-to-even, saturate, dequantize in float64, cast
    back to float32.
    """

    @staticmethod
    def forward(ctx, x, scale64, zero_point, qmin: int, qmax: int):
        code = torch.round(x.double() / scale64) + zero_point
        mask = (code >= qmin) & (code <= qmax)
        ctx.save_for_backward(mask)
        q = torch.clamp(code, qmin, qmax)
        return (scale64 * (q - zero_point)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None, None, None, None


def _broadcast_shape(params: QuantParams, ndim: int) -> list[int]:
    assert params.channel_axis is not None
    shape = [1] * ndim
    shape[params.channel_axis % ndim] = int(params.scale.shape[0])
    return shape


def torch_fake_quantize(x: torch.Tensor, params: QuantParams) -> torch.Tensor:
    """Differentiable fake quantization from frozen QuantScope qparams.

    Mirrors ``affine.fake_quantize`` (same rounding, saturation, and
    float64 arithmetic against float32 scales) with clipped-STE
    gradients. Raises actionable errors for incompatible parameters.
    """
    if params.granularity is Granularity.PER_CHANNEL:
        axis = params.channel_axis % x.dim()  # type: ignore[operator]
        if x.shape[axis] != params.scale.shape[0]:
            raise ValueError(
                f"tensor has {x.shape[axis]} channels on axis {axis}, "
                f"params expect {params.scale.shape[0]}"
            )
        shape = _broadcast_shape(params, x.dim())
        scale64 = torch.as_tensor(params.scale, dtype=torch.float64).reshape(shape)
        zero_point = torch.as_tensor(params.zero_point, dtype=torch.float64).reshape(shape)
    else:
        scale64 = torch.as_tensor(float(params.scale), dtype=torch.float64)
        zero_point = torch.as_tensor(float(params.zero_point), dtype=torch.float64)
    return _FakeQuantSTE.apply(x, scale64, zero_point, params.qmin, params.qmax)


class _WeightFakeQuant(nn.Module):
    """Parametrization: per-channel symmetric weight fake quantization.

    The frozen *rule* of ADR-013: scales are re-derived from the
    current weights every forward (min-max per output channel, the
    same convention as ``compute_quant_params``), detached from
    autograd, then applied through the clipped STE.
    """

    def __init__(self, *, bits: int) -> None:
        super().__init__()
        int_range = integer_range(bits, signed=True)
        self.qmin = int_range.qmin
        self.qmax = int_range.qmax
        self.half_range = float(int_range.qmax)  # quantscope policy: qmax - zp, zp = 0

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(1, weight.dim()))
        bound = weight.detach().double().abs().amax(dim=reduce_dims)
        scale32 = torch.where(bound <= _EPS, torch.ones_like(bound), bound / self.half_range).to(
            torch.float32
        )
        shape = [1] * weight.dim()
        shape[0] = weight.shape[0]
        scale64 = scale32.double().reshape(shape)
        zero_point = torch.zeros_like(scale64)
        return _FakeQuantSTE.apply(weight, scale64, zero_point, self.qmin, self.qmax)


@dataclass(frozen=True)
class QATRecipe:
    """One fine-tuning recipe (ADR-013: only learning_rate varies)."""

    learning_rate: float
    epochs: int = 10
    batch_size: int = 64
    weight_decay: float = 1e-4
    weight_bits: int = 4
    seed: int = 0

    def label(self) -> str:
        return f"lr{self.learning_rate:g}_ep{self.epochs}"


@dataclass
class QATHistory:
    """Diagnostic per-epoch record (never used for checkpoint selection)."""

    epoch_train_loss: list[float] = field(default_factory=list)
    gradients_finite: bool = True


def _attach_frozen_activation_fq(
    model: nn.Module, act_params: Mapping[str, QuantParams]
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach frozen-qparam STE fake quantization at every policy-v1 site."""
    required = set(_all_relu_sites(model)) | {_INPUT_KEY}
    if set(act_params) != required:
        missing = required - set(act_params)
        extra = set(act_params) - required
        raise ValueError(
            f"act_params must cover exactly the policy-v1 sites; missing={sorted(missing)}"
            f" extra={sorted(extra)}"
        )
    for site, params in act_params.items():
        if params.granularity is not Granularity.PER_TENSOR:
            raise ValueError(f"activation site {site!r}: per-tensor qparams required")

    modules = dict(model.named_modules())
    handles = []
    for site, params in act_params.items():
        if site == _INPUT_KEY:
            continue

        def hook(_mod, _inp, out, *, _params=params):
            return torch_fake_quantize(out, _params)

        handles.append(modules[site].register_forward_hook(hook))

    input_params = act_params[_INPUT_KEY]
    handles.append(
        model.register_forward_pre_hook(
            lambda _mod, inp: (torch_fake_quantize(inp[0], input_params), *inp[1:])
        )
    )
    return handles


def _train_loop(model: nn.Module, train_dataset: Dataset, recipe: QATRecipe) -> QATHistory:
    """The shared fine-tuning loop (identical seeding, batching,
    optimizer, and schedule for the QAT arm and the ADR-016 FP32
    control arm — the ONLY difference between arms is whether fake
    quantization is attached to the model before this runs)."""
    torch.manual_seed(recipe.seed)
    generator = torch.Generator().manual_seed(recipe.seed)
    loader = DataLoader(
        train_dataset, batch_size=recipe.batch_size, shuffle=True, generator=generator
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs)
    criterion = nn.CrossEntropyLoss()

    history = QATHistory()
    for epoch in range(recipe.epochs):
        total, count = 0.0, 0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                history.gradients_finite = False
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                    history.gradients_finite = False
                    raise FloatingPointError(f"non-finite gradient in {name} at epoch {epoch}")
            optimizer.step()
            total += float(loss.detach())
            count += 1
        scheduler.step()
        history.epoch_train_loss.append(total / max(count, 1))
        logger.info(
            "finetune %s epoch %d/%d: train loss %.4f",
            recipe.label(),
            epoch + 1,
            recipe.epochs,
            history.epoch_train_loss[-1],
        )
    return history


def qat_finetune(
    model: nn.Module,
    train_dataset: Dataset,
    act_params: Mapping[str, QuantParams],
    recipe: QATRecipe,
) -> tuple[nn.Module, QATHistory]:
    """Fixed-quantization-specification QAT fine-tune (ADR-013).

    Returns ``(finetuned_fp32_model, history)``. The returned model
    carries plain FP32 weights (parametrizations removed); the caller
    exports/evaluates it through the same NumPy simulation pipeline as
    PTQ (``simulate_quantized_with_params``), which recomputes the
    final weight qparams once.

    Raises on any non-finite loss or gradient (success criterion 4 is
    checked during training, not after).
    """
    model = copy.deepcopy(model)
    model.train()

    weight_modules = [m for m in model.modules() if isinstance(m, _WEIGHT_MODULES)]
    if not weight_modules:
        raise ValueError("model has no Conv2d/Linear modules to fine-tune")
    for module in weight_modules:
        parametrize.register_parametrization(
            module, "weight", _WeightFakeQuant(bits=recipe.weight_bits)
        )
    handles = _attach_frozen_activation_fq(model, act_params)

    history = _train_loop(model, train_dataset, recipe)

    for handle in handles:
        handle.remove()
    for module in weight_modules:
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
    model.eval()
    return model, history


def fp32_finetune(
    model: nn.Module, train_dataset: Dataset, recipe: QATRecipe
) -> tuple[nn.Module, QATHistory]:
    """ADR-016 Part A control arm: the identical fine-tune with NO fake
    quantization anywhere — same recipe, seeding, batch order,
    optimizer, and schedule as :func:`qat_finetune`, isolating the
    effect of training through the quantizer from the effect of
    training at all."""
    model = copy.deepcopy(model)
    model.train()
    history = _train_loop(model, train_dataset, recipe)
    model.eval()
    return model, history
