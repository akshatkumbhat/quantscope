"""Integration test: FP32 training -> FX PTQ -> measured INT8 evaluation."""

from pathlib import Path

import pytest

from quantscope.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from quantscope.evaluation import train_fp32
from quantscope.quantization.ptq import run_ptq
from quantscope.utilities import read_metrics


@pytest.mark.integration
def test_fp32_then_ptq_end_to_end(tmp_path: Path) -> None:
    config = ExperimentConfig(
        run_name="e2e",
        output_dir=tmp_path,
        data=DataConfig(num_train=96, num_eval=48, image_size=12),
        model=ModelConfig(num_classes=3),
        training=TrainingConfig(epochs=3, batch_size=16, seed=0),
    )
    _, fp32_metrics = train_fp32(config)
    _quantized, int8_metrics = run_ptq(config)

    # INT8 accuracy must stay close to FP32 on this easy task.
    assert int8_metrics["accuracy"] >= fp32_metrics["accuracy"] - 0.15

    loaded = read_metrics(tmp_path / "e2e-ptq")
    by_name = {m["name"]: m for m in loaded["metrics"]}
    assert by_name["eval_accuracy_int8"]["provenance"] == "measured"
    # INT8 weights serialize smaller than FP32. For a model this tiny,
    # per-tensor serialization overhead (metadata, scales, zero points)
    # dominates, so the ratio is well below the asymptotic ~4x.
    assert by_name["size_compression_ratio"]["value"] > 1.2
    assert (tmp_path / "e2e-ptq" / "model_int8.pt").exists()


@pytest.mark.integration
def test_ptq_without_fp32_checkpoint_fails_clearly(tmp_path: Path) -> None:
    config = ExperimentConfig(run_name="orphan", output_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="fp32 workflow"):
        run_ptq(config)
