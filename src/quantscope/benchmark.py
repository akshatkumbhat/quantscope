"""Texture-10 benchmark A: FP32 vs simulated W8A8/W4A4 discrimination.

Slow path (minutes, not seconds): kept out of the core test suite. Produces
one labeled run artifact per seed with FP32 (measured) and W8A8/W4A4
(simulated) accuracy/NLL/margin.
"""

from __future__ import annotations

import logging

from torch import nn

from quantscope.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    Provenance,
    TrainingConfig,
)
from quantscope.data.texture10 import Texture10Params, make_texture10
from quantscope.evaluation.loop import evaluate_detailed, train_fp32
from quantscope.quantization.simulate import SimQuantConfig, simulate_quantized
from quantscope.utilities import RunWriter

__all__ = ["benchmark_config", "run_texture_benchmark"]

logger = logging.getLogger(__name__)

SIM_CONFIGS = (SimQuantConfig(8, 8), SimQuantConfig(4, 4))


def benchmark_config(
    *,
    seed: int,
    epochs: int = 35,
    boundary_fraction: float = 0.45,
    boundary_low: float = 0.40,
    boundary_high: float = 0.50,
    snr_db: float = 4.0,
    bottleneck_width: int = 6,
    output_dir: str = "runs",
) -> ExperimentConfig:
    """Benchmark-A recipe. Defaults are the FROZEN iteration-3 configuration
    (see ADR-008 addendum): accepted with FP32 ~2pp above the 88-94% target
    band as a documented deviation. Do not retune before plan step B."""
    return ExperimentConfig(
        run_name=f"texture-a-seed{seed}",
        output_dir=output_dir,  # type: ignore[arg-type]
        model=ModelConfig(
            name="bottleneck_resnet", num_classes=10, bottleneck_width=bottleneck_width
        ),
        data=DataConfig(
            name="texture10",
            num_train=5000,
            num_eval=2000,
            num_calib=256,
            image_size=32,
            seed=seed,
            boundary_fraction=boundary_fraction,
            boundary_low=boundary_low,
            boundary_high=boundary_high,
            snr_db=snr_db,
        ),
        training=TrainingConfig(
            epochs=epochs,
            batch_size=64,
            learning_rate=3e-3,
            optimizer="adamw",
            weight_decay=1e-4,
            schedule="cosine",
            seed=seed,
        ),
    )


def texture10_calibration(config: ExperimentConfig):
    """The benchmark's calibration split (disjoint seed stream +2)."""
    return make_texture10(
        num_samples=config.data.num_calib,
        seed=config.data.seed + 2,
        params=Texture10Params(
            num_classes=config.model.num_classes,
            image_size=config.data.image_size,
            boundary_fraction=config.data.boundary_fraction,
            boundary_low=config.data.boundary_low,
            boundary_high=config.data.boundary_high,
            snr_db=config.data.snr_db,
        ),
    )


def run_texture_benchmark(config: ExperimentConfig) -> dict[str, dict[str, float]]:
    """Train FP32, simulate W8A8/W4A4, and write one labeled artifact.

    Returns {"fp32": {...}, "W8A8": {...}, "W4A4": {...}} detailed metrics.
    """
    model, _ = train_fp32(config)
    calibration = texture10_calibration(config)
    # Same test set the fp32 run evaluated on (seed stream +1).
    from quantscope.data.synthetic import build_datasets

    _, test_set = build_datasets(config.data, config.model)

    results: dict[str, dict[str, float]] = {"fp32": evaluate_detailed(model, test_set)}
    writer = RunWriter(config, kind="texture-a")
    for metric, value in results["fp32"].items():
        writer.record_metric(f"fp32_{metric}", value, Provenance.MEASURED)

    for sim in SIM_CONFIGS:
        quantized: nn.Module = simulate_quantized(
            model, calibration, sim, batch_size=config.training.batch_size
        )
        detailed = evaluate_detailed(quantized, test_set)
        results[sim.label] = detailed
        for metric, value in detailed.items():
            writer.record_metric(
                f"{sim.label}_{metric}",
                value,
                Provenance.SIMULATED,
                note="fake-quant simulation policy v1; not integer execution",
            )
        writer.record_metric(
            f"{sim.label}_accuracy_drop",
            results["fp32"]["accuracy"] - detailed["accuracy"],
            Provenance.SIMULATED,
        )
        writer.record_metric(
            f"{sim.label}_nll_increase",
            detailed["nll"] - results["fp32"]["nll"],
            Provenance.SIMULATED,
        )
    writer.finalize()
    logger.info(
        "texture-a seed=%d: fp32=%.3f W8A8=%.3f W4A4=%.3f",
        config.data.seed,
        results["fp32"]["accuracy"],
        results["W8A8"]["accuracy"],
        results["W4A4"]["accuracy"],
    )
    return results
