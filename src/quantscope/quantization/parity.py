"""Staged backend-parity runner (ADR-011, plan step C).

Comparison ladder: sim_backend_matched ↔ reference_fx ↔ real_int8, all
built from ONE calibrated prepare_fx model (same checkpoint, calibration
sample order, eval set, fusion, and observer statistics).

Stages keep failures localized:
  1. qparam parity (all activation sites + every weight channel)
  2. primitive parity on captured tensors (plus the unit-test cases)
  3. activation-only model parity (QuantScope vs torch arithmetic)
  4. weight-only model parity
  5. full graph-anchored sim vs convert_to_reference_fx
  then reference_fx vs convert_fx (real INT8, measured execution)

Stages 3-4 use torch's fake-quant *ops* on the torch side as diagnostic
references; the primary strict gate is stage 5 (per ADR-011, torch's
FakeQuantize module is never used to stand in for QuantScope).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.quantize_fx import convert_fx, convert_to_reference_fx
from torch.utils.data import DataLoader, Dataset

from quantscope.analysis import compare as compare_tensors
from quantscope.config import ExperimentConfig, Provenance
from quantscope.evaluation.loop import evaluate_detailed
from quantscope.quantization.backend_matched import (
    build_graph_anchored_sim,
    extract_histogram_bounds,
    extract_weight_bounds,
    quantscope_params_from_bounds,
    weighted_modules,
)
from quantscope.utilities import RunWriter

__all__ = ["run_backend_parity"]

logger = logging.getLogger(__name__)


@torch.no_grad()
def _collect_logits(model: nn.Module, dataset: Dataset, *, batch_size: int = 64) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return np.concatenate([model(images).numpy() for images, _ in loader])


def _model_comparison(
    name_a: str, logits_a: np.ndarray, name_b: str, logits_b: np.ndarray
) -> dict[str, float]:
    metrics = compare_tensors(logits_b, logits_a).to_dict()
    disagreement = float(np.mean(logits_a.argmax(1) != logits_b.argmax(1)))
    per_sample = np.abs(logits_a - logits_b).max(axis=1)
    return {
        "pair": f"{name_a} vs {name_b}",
        "prediction_disagreement": disagreement,
        "per_sample_max_absdiff_p50": float(np.percentile(per_sample, 50)),
        "per_sample_max_absdiff_p99": float(np.percentile(per_sample, 99)),
        "per_sample_max_absdiff_max": float(per_sample.max()),
        **metrics,
    }


class _TorchFakeQuantOp(nn.Module):
    """Diagnostic-side activation fake-quant using torch's own op."""

    def __init__(self, scale: float, zero_point: int, qmin: int, qmax: int) -> None:
        super().__init__()
        self.scale, self.zero_point, self.qmin, self.qmax = scale, zero_point, qmin, qmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fake_quantize_per_tensor_affine(
            x, self.scale, self.zero_point, self.qmin, self.qmax
        )


def _torch_activation_only(prepared: nn.Module) -> nn.Module:
    model = copy.deepcopy(prepared).eval()
    for name, module in list(model.named_modules()):
        if isinstance(module, HistogramObserver):
            scale, zp = module.calculate_qparams()
            replacement = _TorchFakeQuantOp(
                float(scale), int(zp), int(module.quant_min), int(module.quant_max)
            )
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, replacement)
    return model


def _strip_observers(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model).eval()
    for name, module in list(model.named_modules()):
        if isinstance(module, HistogramObserver):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, nn.Identity())
    return model


def _torch_weight_only(prepared: nn.Module) -> nn.Module:
    from quantscope.quantization.backend_matched import _module_weight

    model = _strip_observers(prepared)
    for _name, module in weighted_modules(model).items():
        observer = module.qconfig.weight()
        weight = _module_weight(module)
        observer(weight.detach())
        scale, zp = observer.calculate_qparams()
        weight.data = torch.fake_quantize_per_channel_affine(
            weight.detach(),
            scale,
            zp.to(torch.int32),  # op requires Int32/Float zero-points
            int(observer.ch_axis),
            int(observer.quant_min),
            int(observer.quant_max),
        )
    return model


def _qparam_parity(prepared: nn.Module) -> list[dict[str, object]]:
    """Stage 1: torch qparams vs QuantScope qparams from identical bounds."""
    rows: list[dict[str, object]] = []
    for name, module in prepared.named_modules():
        if isinstance(module, HistogramObserver):
            t_scale, t_zp = module.calculate_qparams()
            bounds = extract_histogram_bounds(module)
            ours = quantscope_params_from_bounds(bounds, qparam_policy="torch_2_2")
            rows.append(
                {
                    "site": name,
                    "kind": "activation",
                    "torch_scale": float(t_scale),
                    "qs_scale": float(ours.scale),
                    "scale_rel_diff": abs(float(t_scale) - float(ours.scale))
                    / max(float(t_scale), 1e-12),
                    "torch_zp": int(t_zp),
                    "qs_zp": int(ours.zero_point),
                    "zp_equal": int(t_zp) == int(ours.zero_point),
                    "searched_range": [float(bounds.low), float(bounds.high)],
                    "raw_range": [float(bounds.raw_low), float(bounds.raw_high)],
                }
            )
    for name, module in weighted_modules(prepared).items():
        observer = module.qconfig.weight()
        from quantscope.quantization.backend_matched import _module_weight

        observer(_module_weight(module).detach())
        t_scale, t_zp = observer.calculate_qparams()
        bounds = extract_weight_bounds(module)
        compat = quantscope_params_from_bounds(bounds, qparam_policy="torch_2_2")
        native = quantscope_params_from_bounds(bounds, qparam_policy="quantscope")
        t_s, c_s = t_scale.numpy(), np.asarray(compat.scale)
        rows.append(
            {
                "site": name,
                "kind": "weight",
                "channels": int(t_s.size),
                "scale_max_rel_diff_torch22": float(
                    np.max(np.abs(t_s - c_s) / np.maximum(t_s, 1e-12))
                ),
                "zp_all_equal": bool(np.array_equal(t_zp.numpy(), np.asarray(compat.zero_point))),
                # The documented compatibility finding: both calculations shown.
                "native_vs_torch_scale_ratio_mean": float(np.mean(np.asarray(native.scale) / t_s)),
            }
        )
    return rows


def _capture_observer_inputs(
    prepared: nn.Module, calibration: Dataset, *, batch_size: int
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in prepared.named_modules():
        if isinstance(module, HistogramObserver):

            def hook(_m: nn.Module, inp: tuple, _out: object, *, _n: str = name) -> None:
                captured.setdefault(_n, inp[0].detach().clone())

            handles.append(module.register_forward_hook(hook))
    images, _ = next(iter(DataLoader(calibration, batch_size=batch_size, shuffle=False)))
    with torch.no_grad():
        prepared(images)
    for handle in handles:
        handle.remove()
    return captured


def _primitive_parity(prepared: nn.Module, captured: dict[str, torch.Tensor]) -> list[dict]:
    """Stage 2: QuantScope vs torch fake-quant on real captured tensors,
    identical (torch-computed) qparams."""
    from quantscope.quantization.affine import (
        Granularity,
        QuantParams,
        Scheme,
        fake_quantize,
        quantize,
    )

    rows = []
    for name, module in prepared.named_modules():
        if not isinstance(module, HistogramObserver) or name not in captured:
            continue
        tensor = captured[name].numpy().astype(np.float32)
        t_scale, t_zp = module.calculate_qparams()
        params = QuantParams(
            scale=np.asarray(float(t_scale), dtype=np.float32),
            zero_point=np.asarray(int(t_zp), dtype=np.int32),
            qmin=int(module.quant_min),
            qmax=int(module.quant_max),
            bits=7,
            signed=False,
            scheme=Scheme.ASYMMETRIC,
            granularity=Granularity.PER_TENSOR,
        )
        torch_fq = torch.fake_quantize_per_tensor_affine(
            captured[name],
            float(t_scale),
            int(t_zp),
            int(module.quant_min),
            int(module.quant_max),
        ).numpy()
        ours_fq = fake_quantize(tensor, params)
        code_diff = int(
            np.sum(
                quantize(tensor, params)
                != np.round(torch_fq / float(t_scale) + int(t_zp)).astype(np.int32)
            )
        )
        rows.append(
            {
                "site": name,
                "elements": int(tensor.size),
                "fq_max_absdiff": float(np.max(np.abs(ours_fq - torch_fq))),
                "code_mismatches": code_diff,
            }
        )
    return rows


def _graph_summary(model: nn.Module, label: str) -> dict[str, object]:
    graph = getattr(model, "graph", None)
    if graph is None:
        return {"label": label, "graph": "not an FX graph module"}
    ops: dict[str, int] = {}
    quantize_nodes = 0
    float_function_nodes: list[str] = []
    for node in graph.nodes:
        key = f"{node.op}:{node.target}" if node.op == "call_function" else node.op
        ops[key] = ops.get(key, 0) + 1
        target_str = str(node.target)
        if "quantize" in target_str:
            quantize_nodes += 1
        elif node.op == "call_function" and "dequantize" not in target_str:
            float_function_nodes.append(f"{node.name}:{target_str}")
    return {
        "label": label,
        "num_nodes": sum(ops.values()),
        "quantize_dequantize_nodes": quantize_nodes,
        "float_function_nodes": float_function_nodes,
        "graph_text": str(graph),
    }


def run_backend_parity(
    config: ExperimentConfig,
    calibration: Dataset,
    test_set: Dataset,
    *,
    checkpoint: str | Path,
) -> dict[str, object]:
    """Run the full staged parity investigation on one checkpoint."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx

    from quantscope.models.tiny_cnn import build_model

    torch.backends.quantized.engine = "fbgemm"
    model = build_model(config.model)
    model.load_state_dict(torch.load(Path(checkpoint)))
    model.eval()

    example = (test_set[0][0].unsqueeze(0),)
    prepared = prepare_fx(model, get_default_qconfig_mapping("fbgemm"), example)
    loader = DataLoader(calibration, batch_size=config.training.batch_size, shuffle=False)
    with torch.no_grad():
        for images, _ in loader:
            prepared(images)

    writer = RunWriter(config, kind="backend-parity")
    results: dict[str, object] = {}

    # Stage 1: qparam parity.
    qparams = _qparam_parity(prepared)
    results["stage1_qparams"] = qparams

    # Stage 2: primitive parity on captured tensors.
    captured = _capture_observer_inputs(
        prepared, calibration, batch_size=config.training.batch_size
    )
    results["stage2_primitives"] = _primitive_parity(prepared, captured)

    # Build all model variants from the SAME calibrated prepared model.
    sim_full, sim_params = build_graph_anchored_sim(prepared)
    sim_acts, _ = build_graph_anchored_sim(prepared, quantize_weights=False)
    sim_weights, _ = build_graph_anchored_sim(prepared, quantize_activations=False)
    torch_acts = _torch_activation_only(prepared)
    torch_weights = _torch_weight_only(prepared)
    reference = convert_to_reference_fx(copy.deepcopy(prepared))
    real_int8 = convert_fx(copy.deepcopy(prepared))

    logits = {
        "sim_full": _collect_logits(sim_full, test_set),
        "sim_acts_only": _collect_logits(sim_acts, test_set),
        "sim_weights_only": _collect_logits(sim_weights, test_set),
        "torch_acts_only": _collect_logits(torch_acts, test_set),
        "torch_weights_only": _collect_logits(torch_weights, test_set),
        "reference_fx": _collect_logits(reference, test_set),
        "real_int8": _collect_logits(real_int8, test_set),
    }
    results["stage3_activation_only"] = _model_comparison(
        "sim_acts_only", logits["sim_acts_only"], "torch_acts_only", logits["torch_acts_only"]
    )
    results["stage4_weight_only"] = _model_comparison(
        "sim_weights_only",
        logits["sim_weights_only"],
        "torch_weights_only",
        logits["torch_weights_only"],
    )
    results["stage5_strict"] = _model_comparison(
        "sim_full", logits["sim_full"], "reference_fx", logits["reference_fx"]
    )
    results["backend_comparison"] = _model_comparison(
        "reference_fx", logits["reference_fx"], "real_int8", logits["real_int8"]
    )

    for name, model_obj in (
        ("sim_full", sim_full),
        ("reference_fx", reference),
        ("real_int8", real_int8),
    ):
        detailed = evaluate_detailed(model_obj, test_set)
        provenance = Provenance.MEASURED if name == "real_int8" else Provenance.SIMULATED
        for metric, value in detailed.items():
            writer.record_metric(f"{name}_{metric}", value, provenance)
        results[f"{name}_eval"] = detailed

    graphs = [
        _graph_summary(prepared, "prepared"),
        _graph_summary(reference, "reference_fx"),
        _graph_summary(real_int8, "real_int8"),
    ]
    results["graphs"] = [{k: v for k, v in g.items() if k != "graph_text"} for g in graphs]

    # Artifacts: qparam table, stage results, per-sample logit diffs, graphs.
    (writer.run_dir / "parity_results.json").write_text(
        json.dumps(results, indent=1, default=str) + "\n"
    )
    np.save(
        writer.run_dir / "per_sample_logit_absdiff_sim_vs_reference.npy",
        np.abs(logits["sim_full"] - logits["reference_fx"]).max(axis=1),
    )
    for g in graphs:
        (writer.run_dir / f"graph_{g['label']}.txt").write_text(str(g.get("graph_text", "")))
    writer.record_metric(
        "sim_params_recorded",
        len(sim_params),
        Provenance.SIMULATED,
        note="scale/zero-point metadata in parity_results.json",
    )
    writer.finalize()
    return results
