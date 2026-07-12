"""Per-group quantization ablation: quantize one group, measure the damage.

For each group in the partition, the group alone is simulated at a target
bit width (default W4A4) while everything else stays FP32. Sensitivity is
reported as ΔNLL (primary), prediction flips vs. FP32, Δaccuracy, and
Δmargin. All ablation results are **simulated** (policy v1).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from quantscope.config import ExperimentConfig, Provenance
from quantscope.evaluation.loop import evaluate_detailed
from quantscope.models.tiny_cnn import build_model
from quantscope.quantization.simulate import (
    BOTTLENECK_RESNET_GROUPS,
    GroupSpec,
    SimQuantConfig,
    simulate_quantized_groups,
)
from quantscope.utilities import RunWriter

__all__ = ["ablate_groups", "predictions"]

logger = logging.getLogger(__name__)

_DEFAULT_TARGET = SimQuantConfig(4, 4)


@torch.no_grad()
def predictions(model: nn.Module, dataset: Dataset, *, batch_size: int = 64) -> np.ndarray:
    """Top-1 predictions over a dataset (deterministic order)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = [model(images).argmax(dim=1).numpy() for images, _ in loader]
    return np.concatenate(out)


def ablate_groups(
    config: ExperimentConfig,
    calibration: Dataset,
    test_set: Dataset,
    *,
    checkpoint: str | Path,
    target: SimQuantConfig = _DEFAULT_TARGET,
    groups: Mapping[str, GroupSpec] = BOTTLENECK_RESNET_GROUPS,
) -> dict[str, dict[str, float]]:
    """One-group-at-a-time ablation against an FP32 checkpoint.

    Returns per-group deltas and writes a labeled run artifact
    (kind="ablation-<target>").
    """
    model = build_model(config.model)
    model.load_state_dict(torch.load(Path(checkpoint)))
    model.eval()

    fp32_detailed = evaluate_detailed(model, test_set)
    fp32_preds = predictions(model, test_set)

    writer = RunWriter(config, kind=f"ablation-{target.label.lower()}")
    for metric, value in fp32_detailed.items():
        writer.record_metric(f"fp32_{metric}", value, Provenance.MEASURED)

    results: dict[str, dict[str, float]] = {}
    for group_name in groups:
        assignment: dict[str, SimQuantConfig | None] = dict.fromkeys(groups, None)
        assignment[group_name] = target
        quantized = simulate_quantized_groups(
            model,
            calibration,
            assignment,
            groups=groups,
            batch_size=config.training.batch_size,
        )
        detailed = evaluate_detailed(quantized, test_set)
        flips = float(np.mean(predictions(quantized, test_set) != fp32_preds))
        deltas = {
            "delta_nll": detailed["nll"] - fp32_detailed["nll"],
            "prediction_flip_rate": flips,
            "delta_accuracy": detailed["accuracy"] - fp32_detailed["accuracy"],
            "delta_margin": detailed["mean_margin"] - fp32_detailed["mean_margin"],
        }
        results[group_name] = deltas
        for metric, value in deltas.items():
            writer.record_metric(
                f"{group_name}_{metric}",
                value,
                Provenance.SIMULATED,
                note=f"only group '{group_name}' at {target.label}; policy v1",
            )
        logger.info(
            "ablation %s: dNLL=%+.4f flips=%.4f dAcc=%+.4f",
            group_name,
            deltas["delta_nll"],
            flips,
            deltas["delta_accuracy"],
        )
    writer.finalize()
    return results
