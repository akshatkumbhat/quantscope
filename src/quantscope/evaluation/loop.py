"""FP32 training and evaluation loops (CPU).

All metrics produced here are **measured**: they come from actually
executing the model on the given data.
"""

from __future__ import annotations

import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from quantscope.config import ExperimentConfig, Provenance
from quantscope.data.synthetic import build_datasets
from quantscope.models.tiny_cnn import build_model
from quantscope.utilities import RunWriter, set_seed

__all__ = ["evaluate", "train_fp32"]

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, *, batch_size: int = 64) -> dict[str, float]:
    """Measure accuracy and mean loss on a dataset. CPU, deterministic."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        logits = model(images)
        total_loss += float(criterion(logits, labels))
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += labels.numel()
    if count == 0:
        raise ValueError("evaluation dataset is empty")
    return {"accuracy": correct / count, "loss": total_loss / count}


def train_fp32(config: ExperimentConfig) -> tuple[nn.Module, dict[str, float]]:
    """Train the configured model in FP32 and write a labeled run artifact.

    Returns the trained model and its measured eval metrics.
    """
    set_seed(config.training.seed)
    model = build_model(config.model)
    train_set, eval_set = build_datasets(config.data, config.model)

    writer = RunWriter(config, kind="fp32")
    loader = DataLoader(
        train_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    model.train()
    for epoch in range(config.training.epochs):
        epoch_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        logger.info("epoch %d/%d loss=%.4f", epoch + 1, config.training.epochs, epoch_loss)
    train_seconds = time.perf_counter() - start

    metrics = evaluate(model, eval_set)
    writer.record_metric("eval_accuracy", metrics["accuracy"], Provenance.MEASURED)
    writer.record_metric("eval_loss", metrics["loss"], Provenance.MEASURED)
    writer.record_metric(
        "train_wall_seconds",
        train_seconds,
        Provenance.MEASURED,
        note="CPU wall time; not accelerator latency",
    )

    checkpoint = writer.run_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    run_dir = writer.finalize()
    logger.info("fp32 run complete: %s (accuracy=%.3f)", run_dir, metrics["accuracy"])
    return model, metrics
