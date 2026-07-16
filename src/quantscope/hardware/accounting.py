"""Deterministic model accounting for the analytical cost model (ADR-014).

One shape-instrumented forward pass of the FP32 benchmark model records,
per frozen quantization group: parameter counts, MAC counts, and the
distinct quantized activation tensors the group *produces* (policy-v1
sites). Counts are per sample (batch dimension excluded).

Declared traffic assumption (`single-read-single-write-per-tensor-v1`):
no cache; each distinct modeled activation tensor is counted once for
write and once for read at the precision of its producing group, with
no per-consumer multiplier — shared residual tensors are therefore
counted once. The model input is owned by the stem group (one read).
Host transfer and input-quantization overhead are excluded.

Explicit exclusions (constant across configurations, outside every
total): BatchNorm, residual adds, pooling, flatten, the unquantized
output logits, and quantize/dequantize boundaries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn

from quantscope.quantization.simulate import BOTTLENECK_RESNET_GROUPS

__all__ = [
    "GROUP_ORDER_V1",
    "GROUP_ORDER_VERSION",
    "GroupAccount",
    "LayerAccount",
    "ModelAccounting",
    "TensorAccount",
    "account_model",
]

GROUP_ORDER_VERSION = "group-order-v1"
# Explicit, versioned canonical order — NOT incidental dict order. Must
# match the frozen B3 partition exactly (verified in account_model).
GROUP_ORDER_V1: tuple[str, ...] = (
    "stem",
    "block_a_conv1",
    "block_a_conv2",
    "down",
    "block_b",
    "bottleneck",
    "expand",
    "classifier",
)


@dataclass(frozen=True)
class LayerAccount:
    """One weighted (quantizable) module."""

    name: str
    kind: str  # "conv2d" | "linear"
    parameters: int
    macs: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class TensorAccount:
    """One distinct modeled activation tensor, owned by its producer."""

    site: str  # policy-v1 observer site name ("__input__" or a ReLU module)
    elements: int  # per sample
    shape: tuple[int, ...]
    producer_group: str
    traffic: str  # "read" (model input) or "read+write" (produced tensors)


@dataclass(frozen=True)
class GroupAccount:
    name: str
    layers: tuple[LayerAccount, ...]
    tensors: tuple[TensorAccount, ...]

    @property
    def parameters(self) -> int:
        return sum(layer.parameters for layer in self.layers)

    @property
    def macs(self) -> int:
        return sum(layer.macs for layer in self.layers)


@dataclass(frozen=True)
class ModelAccounting:
    """Complete deterministic accounting for one model architecture."""

    group_order_version: str
    input_shape: tuple[int, ...]
    groups: tuple[GroupAccount, ...]  # in GROUP_ORDER_V1 order
    excluded_operations: dict[str, int]
    traffic_model: str

    @property
    def total_parameters(self) -> int:
        return sum(g.parameters for g in self.groups)

    @property
    def total_macs(self) -> int:
        return sum(g.macs for g in self.groups)

    @property
    def modeled_weighted_modules(self) -> int:
        return sum(len(g.layers) for g in self.groups)

    def group(self, name: str) -> GroupAccount:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(f"no accounting group named {name!r}")

    def to_payload(self) -> dict:
        return {
            "group_order_version": self.group_order_version,
            "input_shape": list(self.input_shape),
            "traffic_model": self.traffic_model,
            "excluded_operations": dict(sorted(self.excluded_operations.items())),
            "groups": [asdict(g) for g in self.groups],
            "totals": {
                "parameters": self.total_parameters,
                "macs": self.total_macs,
                "modeled_weighted_modules": self.modeled_weighted_modules,
            },
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_payload(), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()


def _per_sample_elements(shape: tuple[int, ...]) -> int:
    elements = 1
    for dim in shape[1:]:  # batch dimension excluded
        elements *= dim
    return elements


def account_model(model: nn.Module, *, input_shape: tuple[int, ...] = (1, 1, 32, 32)):
    """Build the deterministic accounting record for ``model``.

    Fails loudly if the model's weighted modules or ReLU sites do not
    exactly match the frozen B3 partition.
    """
    if tuple(BOTTLENECK_RESNET_GROUPS) != GROUP_ORDER_V1:
        raise RuntimeError(
            "frozen B3 partition no longer matches GROUP_ORDER_V1 — the canonical "
            "group order is versioned and must not drift silently"
        )

    model = model.eval()
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d | nn.Linear | nn.ReLU):

            def hook(_m, inp, out, *, _name=name):
                shapes[_name] = (tuple(inp[0].shape), tuple(out.shape))

            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for handle in handles:
        handle.remove()

    modules = dict(model.named_modules())

    def layer_account(name: str) -> LayerAccount:
        module = modules.get(name)
        if module is None or name not in shapes:
            raise ValueError(f"weighted module {name!r} missing from model or trace")
        in_shape, out_shape = shapes[name]
        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise ValueError(f"{name}: grouped convolutions are not modeled")
            out_elems = _per_sample_elements(out_shape)
            kh, kw = module.kernel_size
            macs = out_elems * module.in_channels * kh * kw
            kind = "conv2d"
        elif isinstance(module, nn.Linear):
            macs = module.in_features * module.out_features
            kind = "linear"
        else:
            raise ValueError(f"{name}: unsupported weighted module {type(module).__name__}")
        return LayerAccount(
            name=name,
            kind=kind,
            parameters=module.weight.numel(),
            macs=macs,
            input_shape=in_shape,
            output_shape=out_shape,
        )

    groups = []
    seen_sites: set[str] = set()
    for group_name in GROUP_ORDER_V1:
        spec = BOTTLENECK_RESNET_GROUPS[group_name]
        tensors = []
        if spec.include_input:
            tensors.append(
                TensorAccount(
                    site="__input__",
                    elements=_per_sample_elements(input_shape),
                    shape=input_shape,
                    producer_group=group_name,
                    traffic="read",
                )
            )
        for site in spec.activations:
            if site not in shapes:
                raise ValueError(f"activation site {site!r} missing from trace")
            out_shape = shapes[site][1]
            tensors.append(
                TensorAccount(
                    site=site,
                    elements=_per_sample_elements(out_shape),
                    shape=out_shape,
                    producer_group=group_name,
                    traffic="read+write",
                )
            )
            seen_sites.add(site)
        groups.append(
            GroupAccount(
                name=group_name,
                layers=tuple(layer_account(w) for w in spec.weights),
                tensors=tuple(tensors),
            )
        )

    relu_sites = {name for name, m in modules.items() if isinstance(m, nn.ReLU)}
    if relu_sites != seen_sites:
        raise ValueError(
            f"partition activation sites {sorted(seen_sites)} do not exactly match the "
            f"model's ReLU sites {sorted(relu_sites)}"
        )

    excluded = {
        "batchnorm2d": sum(1 for m in modules.values() if isinstance(m, nn.BatchNorm2d)),
        "adaptive_avg_pool2d": sum(
            1 for m in modules.values() if isinstance(m, nn.AdaptiveAvgPool2d)
        ),
        "residual_add": 2,  # block_a and block_b skip connections
        "flatten": 1,
        "output_logits_tensor": 1,  # unquantized float island under policy v1
        "quantize_dequantize_boundaries": len(relu_sites) + 1,
    }
    return ModelAccounting(
        group_order_version=GROUP_ORDER_VERSION,
        input_shape=input_shape,
        groups=tuple(groups),
        excluded_operations=excluded,
        traffic_model="single-read-single-write-per-tensor-v1",
    )
