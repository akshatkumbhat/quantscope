"""Unit tests for reproducibility and artifact utilities."""

from pathlib import Path

import pytest
import torch

from quantscope.config import ExperimentConfig, Provenance
from quantscope.utilities import RunWriter, capture_environment, read_metrics, set_seed


class TestSeeding:
    def test_torch_deterministic(self) -> None:
        set_seed(0)
        a = torch.randn(4)
        set_seed(0)
        b = torch.randn(4)
        assert torch.equal(a, b)


class TestEnvironment:
    def test_capture_keys(self) -> None:
        env = capture_environment()
        assert set(env) >= {"python", "platform", "packages", "device", "git_commit"}
        assert env["packages"]["torch"] != "not-installed"
        assert env["device"] == "cpu"


class TestRunWriter:
    def _config(self, tmp_path: Path) -> ExperimentConfig:
        return ExperimentConfig(run_name="t", output_dir=tmp_path)

    def test_writes_config_env_metrics(self, tmp_path: Path) -> None:
        writer = RunWriter(self._config(tmp_path), kind="fp32")
        writer.record_metric("accuracy", 0.9, Provenance.MEASURED)
        run_dir = writer.finalize()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "environment.json").exists()
        loaded = read_metrics(run_dir)
        assert loaded["kind"] == "fp32"
        assert loaded["metrics"][0]["provenance"] == "measured"

    def test_unlabeled_metric_rejected(self, tmp_path: Path) -> None:
        writer = RunWriter(self._config(tmp_path), kind="fp32")
        with pytest.raises(TypeError, match="Provenance"):
            writer.record_metric("accuracy", 0.9, "measured")  # type: ignore[arg-type]

    def test_read_missing_metrics_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_metrics(tmp_path)
