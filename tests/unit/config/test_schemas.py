"""Unit tests for experiment configuration schemas."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from quantscope.config import (
    ExperimentConfig,
    QuantizationConfig,
    load_experiment_config,
)


class TestExperimentConfig:
    def test_defaults_valid(self) -> None:
        cfg = ExperimentConfig(run_name="baseline")
        assert cfg.model.name == "tiny_cnn"
        assert cfg.quantization.bits == 8

    def test_run_name_validated(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(run_name="bad name with spaces!")

    def test_extra_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(run_name="x", unknown_field=1)  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        cfg = ExperimentConfig(run_name="x")
        with pytest.raises(ValidationError):
            cfg.run_name = "y"  # type: ignore[misc]

    def test_invalid_bits_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QuantizationConfig(bits=1)
        with pytest.raises(ValidationError):
            QuantizationConfig(bits=32)

    def test_inverted_percentiles_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lower_percentile"):
            QuantizationConfig(lower_percentile=99.0, upper_percentile=1.0)


class TestLoadYaml:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "exp.yaml"
        path.write_text(
            "run_name: quick\n"
            "training:\n  epochs: 2\n  seed: 7\n"
            "quantization:\n  observer: percentile\n"
        )
        cfg = load_experiment_config(path)
        assert cfg.run_name == "quick"
        assert cfg.training.epochs == 2
        assert cfg.quantization.observer == "percentile"

    def test_non_mapping_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="mapping"):
            load_experiment_config(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_experiment_config(tmp_path / "nope.yaml")
