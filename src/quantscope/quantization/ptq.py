"""Post-training quantization via FX graph mode (ADR-006).

The converted model executes with real INT8 CPU kernels (`fbgemm`/`x86`
engine), so its accuracy is a **measured** value. Its serialized size is
also measured. Nothing here estimates accelerator latency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader, Dataset

from quantscope.config import ExperimentConfig, Provenance
from quantscope.data.synthetic import build_datasets
from quantscope.evaluation.loop import evaluate
from quantscope.models.tiny_cnn import build_model
from quantscope.utilities import RunWriter, set_seed

__all__ = ["run_ptq"]

logger = logging.getLogger(__name__)

_ENGINE = "fbgemm"


def _serialized_size_bytes(model: nn.Module, path: Path) -> int:
    torch.save(model.state_dict(), path)
    return path.stat().st_size


def _calibrate(prepared: nn.Module, dataset: Dataset, *, batches: int, batch_size: int) -> int:
    """Run calibration batches through the prepared (observed) model."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    seen = 0
    with torch.no_grad():
        for images, _ in loader:
            prepared(images)
            seen += 1
            if seen >= batches:
                break
    if seen == 0:
        raise ValueError("calibration dataset produced no batches")
    return seen


def run_ptq(
    config: ExperimentConfig, *, checkpoint: str | Path | None = None
) -> tuple[nn.Module, dict[str, float]]:
    """Quantize an FP32 model with FX-graph-mode PTQ and measure it.

    Loads the FP32 checkpoint from the matching fp32 run (or an explicit
    ``checkpoint`` path), calibrates on training data, converts to a real
    INT8 CPU model, and records measured accuracy/size plus the FP32
    comparison in a labeled run artifact.

    Raises:
        FileNotFoundError: if no FP32 checkpoint exists yet.
    """
    torch.backends.quantized.engine = _ENGINE
    set_seed(config.training.seed)

    if checkpoint is None:
        checkpoint = Path(config.output_dir) / f"{config.run_name}-fp32" / "model.pt"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"FP32 checkpoint not found at {checkpoint}; run the fp32 workflow first"
        )

    model = build_model(config.model)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    train_set, eval_set = build_datasets(config.data, config.model)
    fp32_metrics = evaluate(model, eval_set)

    writer = RunWriter(config, kind="ptq")
    example_inputs = (train_set[0][0].unsqueeze(0),)
    qconfig_mapping = get_default_qconfig_mapping(_ENGINE)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    batches = _calibrate(
        prepared,
        train_set,
        batches=config.quantization.calibration_batches,
        batch_size=config.training.batch_size,
    )
    quantized = convert_fx(prepared)

    int8_metrics = evaluate(quantized, eval_set)
    fp32_size = _serialized_size_bytes(model, writer.run_dir / "model_fp32_ref.pt")
    int8_size = _serialized_size_bytes(quantized, writer.run_dir / "model_int8.pt")

    writer.record_metric(
        "eval_accuracy_int8",
        int8_metrics["accuracy"],
        Provenance.MEASURED,
        note=f"real INT8 CPU execution ({_ENGINE} engine)",
    )
    writer.record_metric("eval_accuracy_fp32", fp32_metrics["accuracy"], Provenance.MEASURED)
    writer.record_metric(
        "accuracy_delta",
        int8_metrics["accuracy"] - fp32_metrics["accuracy"],
        Provenance.MEASURED,
    )
    writer.record_metric("calibration_batches", batches, Provenance.MEASURED)
    writer.record_metric("model_size_bytes_fp32", fp32_size, Provenance.MEASURED)
    writer.record_metric("model_size_bytes_int8", int8_size, Provenance.MEASURED)
    writer.record_metric("size_compression_ratio", fp32_size / int8_size, Provenance.MEASURED)
    run_dir = writer.finalize()
    logger.info(
        "ptq run complete: %s (int8 accuracy=%.3f, fp32 accuracy=%.3f)",
        run_dir,
        int8_metrics["accuracy"],
        fp32_metrics["accuracy"],
    )
    return quantized, int8_metrics
