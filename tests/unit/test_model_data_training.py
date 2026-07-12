"""Unit tests for the tiny model, synthetic data, and training loop."""

from pathlib import Path

import numpy as np
import pytest
import torch

from quantscope.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from quantscope.data import make_synthetic_dataset
from quantscope.evaluation import evaluate, train_fp32
from quantscope.models import TinyCNN, build_model
from quantscope.utilities import read_metrics


class TestTinyCNN:
    def test_forward_shape(self) -> None:
        model = TinyCNN(num_classes=4, in_channels=1)
        out = model(torch.randn(2, 1, 16, 16))
        assert out.shape == (2, 4)

    def test_fx_traceable(self) -> None:
        # FX symbolic tracing is a hard prerequisite for FX-mode PTQ (ADR-006).
        model = TinyCNN().eval()
        traced = torch.fx.symbolic_trace(model)
        out = traced(torch.randn(1, 1, 16, 16))
        assert out.shape == (1, 4)

    def test_unknown_model_rejected(self) -> None:
        cfg = ModelConfig()
        object.__setattr__(cfg, "name", "resnet50")  # bypass frozen for the test
        with pytest.raises(ValueError, match="unknown model"):
            build_model(cfg)


class TestSyntheticData:
    def test_shapes_and_labels(self) -> None:
        ds = make_synthetic_dataset(
            num_samples=32, image_size=16, num_classes=4, in_channels=1, seed=0
        )
        images, labels = ds.tensors
        assert images.shape == (32, 1, 16, 16)
        assert images.dtype == torch.float32
        assert set(labels.tolist()) == {0, 1, 2, 3}

    def test_deterministic(self) -> None:
        kwargs = {"num_samples": 16, "image_size": 8, "num_classes": 2, "in_channels": 1}
        a = make_synthetic_dataset(seed=5, **kwargs)
        b = make_synthetic_dataset(seed=5, **kwargs)
        assert torch.equal(a.tensors[0], b.tensors[0])

    def test_different_seeds_differ(self) -> None:
        kwargs = {"num_samples": 16, "image_size": 8, "num_classes": 2, "in_channels": 1}
        a = make_synthetic_dataset(seed=1, **kwargs)
        b = make_synthetic_dataset(seed=2, **kwargs)
        assert not torch.equal(a.tensors[0], b.tensors[0])

    def test_classes_separable(self) -> None:
        # Class means must differ far more than noise: a sanity check that
        # the task is genuinely learnable.
        ds = make_synthetic_dataset(
            num_samples=64, image_size=16, num_classes=2, in_channels=1, seed=0
        )
        images, labels = ds.tensors
        m0 = images[labels == 0].mean(dim=0).numpy()
        m1 = images[labels == 1].mean(dim=0).numpy()
        assert np.abs(m0 - m1).max() > 0.5

    def test_too_few_samples_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_samples"):
            make_synthetic_dataset(
                num_samples=2, image_size=8, num_classes=4, in_channels=1, seed=0
            )


def _quick_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="quick-test",
        output_dir=tmp_path,
        data=DataConfig(num_train=96, num_eval=48, image_size=12),
        model=ModelConfig(num_classes=3),
        training=TrainingConfig(epochs=3, batch_size=16, seed=0),
    )


class TestTraining:
    def test_train_fp32_learns_and_writes_artifacts(self, tmp_path: Path) -> None:
        config = _quick_config(tmp_path)
        _model, metrics = train_fp32(config)
        # Better than chance (1/3) with margin: the model actually learned.
        assert metrics["accuracy"] > 0.5
        run_dir = tmp_path / "quick-test-fp32"
        assert (run_dir / "model.pt").exists()
        loaded = read_metrics(run_dir)
        names = {m["name"] for m in loaded["metrics"]}
        assert {"eval_accuracy", "eval_loss", "train_wall_seconds"} <= names
        assert all(m["provenance"] == "measured" for m in loaded["metrics"])

    def test_evaluate_empty_dataset_rejected(self) -> None:
        model = TinyCNN()
        empty = torch.utils.data.TensorDataset(
            torch.empty(0, 1, 16, 16), torch.empty(0, dtype=torch.int64)
        )
        with pytest.raises(ValueError, match="empty"):
            evaluate(model, empty)
