"""Typed experiment configuration schemas (pydantic v2).

Configs are data: loadable from YAML, fully validated, and serialized into
every run directory so experiments are reproducible from artifacts alone.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantscope.quantization import MAX_BITS, MIN_BITS, Granularity, Scheme

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "ObserverKind",
    "Provenance",
    "QuantizationConfig",
    "TrainingConfig",
    "load_experiment_config",
]


class Provenance(StrEnum):
    """How a reported value was obtained (ADR-004). Every persisted metric
    carries one of these labels; nothing is reported unlabeled."""

    MEASURED = "measured"
    SIMULATED = "simulated"
    ESTIMATED = "estimated"


class ObserverKind(StrEnum):
    """Calibration observer selection."""

    MINMAX = "minmax"
    PERCENTILE = "percentile"
    POWER_OF_TWO = "power_of_two"
    MSE_GRID = "mse_grid"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_StrictModel):
    """Model architecture selection."""

    name: Literal["tiny_cnn", "bottleneck_resnet"] = "tiny_cnn"
    num_classes: int = Field(default=4, ge=2, le=1000)
    in_channels: int = Field(default=1, ge=1, le=4)
    bottleneck_width: int = Field(default=6, ge=2, le=16)  # bottleneck_resnet only


class DataConfig(_StrictModel):
    """Dataset selection. Both datasets are synthetic: no downloads/network."""

    name: Literal["synthetic", "texture10"] = "synthetic"
    num_train: int = Field(default=512, ge=8)
    num_eval: int = Field(default=256, ge=8)
    num_calib: int = Field(default=256, ge=8)  # texture10 only
    image_size: int = Field(default=16, ge=8, le=64)
    seed: int = 0
    # texture10 difficulty knobs (ignored by `synthetic`).
    boundary_fraction: float = Field(default=0.2, ge=0.0, le=0.5)
    # Interpolation range toward the neighbor class; 0.5 = fully ambiguous
    # mixture (irreducible error on those samples by design).
    boundary_low: float = Field(default=0.30, ge=0.0, le=0.5)
    boundary_high: float = Field(default=0.45, ge=0.3, le=0.5)

    @model_validator(mode="after")
    def _check_boundary(self) -> DataConfig:
        if self.boundary_low >= self.boundary_high:
            raise ValueError(
                f"boundary_low ({self.boundary_low}) must be < boundary_high ({self.boundary_high})"
            )
        return self

    snr_db: float = Field(default=8.0, ge=0.0, le=30.0)


class TrainingConfig(_StrictModel):
    """FP32 training hyperparameters (CPU-friendly by design)."""

    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-2, gt=0)
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    schedule: Literal["none", "cosine"] = "none"
    seed: int = 0


class QuantizationConfig(_StrictModel):
    """PTQ/QAT settings."""

    bits: int = Field(default=8, ge=MIN_BITS, le=MAX_BITS)
    signed: bool = True
    scheme: Scheme = Scheme.ASYMMETRIC
    granularity: Granularity = Granularity.PER_TENSOR
    observer: ObserverKind = ObserverKind.MINMAX
    # Percentile-observer settings (used when observer == PERCENTILE).
    lower_percentile: float = Field(default=0.1, ge=0.0, lt=100.0)
    upper_percentile: float = Field(default=99.9, gt=0.0, le=100.0)
    calibration_batches: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _check_percentiles(self) -> QuantizationConfig:
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError(
                f"lower_percentile ({self.lower_percentile}) must be < "
                f"upper_percentile ({self.upper_percentile})"
            )
        return self


class ExperimentConfig(_StrictModel):
    """Top-level experiment description."""

    run_name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_\-]+$")
    output_dir: Path = Path("runs")
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    quantization: QuantizationConfig = QuantizationConfig()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the YAML is not a mapping.
        pydantic.ValidationError: if validation fails.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"expected a YAML mapping in {path}, got {type(raw).__name__}")
    return ExperimentConfig.model_validate(raw)
