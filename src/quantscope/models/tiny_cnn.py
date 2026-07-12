"""Tiny FX-traceable CNN for CPU-fast quantization experiments."""

from __future__ import annotations

import torch
from torch import nn

from quantscope.config import ModelConfig

__all__ = ["TinyCNN", "build_model"]


class TinyCNN(nn.Module):
    """Two conv blocks + linear head.

    Deliberately small (trains on CPU in seconds) yet structurally
    representative: conv/bn/relu fusion patterns, pooling, and a linear
    classifier — the shapes FX-graph-mode quantization must handle.
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(config: ModelConfig) -> nn.Module:
    """Build the configured model architecture."""
    if config.name == "tiny_cnn":
        return TinyCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    if config.name == "bottleneck_resnet":
        from quantscope.models.bottleneck_resnet import BottleneckResNet

        return BottleneckResNet(
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            bottleneck_width=config.bottleneck_width,
        )
    raise ValueError(f"unknown model: {config.name!r}")
