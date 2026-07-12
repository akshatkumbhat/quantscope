"""Narrow residual CNN with a deliberate information bottleneck.

~20k parameters. Structurally heterogeneous quantization sites by design
(stem / residual branches / downsample / narrow bottleneck / expansion /
classifier) so per-layer quantization sensitivity has a reason to be
non-uniform. FX-traceable; every ReLU is a distinct module instance so
activation hooks and quantization group mapping stay unambiguous.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["BottleneckResNet"]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu_out(out + x)


class BottleneckResNet(nn.Module):
    """Stem -> ResA -> Downsample -> ResB -> bottleneck -> expansion -> head."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        bottleneck_width: int = 6,
    ) -> None:
        super().__init__()
        if bottleneck_width < 2:
            raise ValueError("bottleneck_width must be >= 2")
        self.stem_conv = nn.Conv2d(in_channels, 16, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_relu = nn.ReLU()

        self.block_a = _ResidualBlock(16)

        self.down_conv = nn.Conv2d(16, 24, 3, stride=2, padding=1, bias=False)
        self.down_bn = nn.BatchNorm2d(24)
        self.down_relu = nn.ReLU()

        self.block_b = _ResidualBlock(24)

        self.bottleneck_conv = nn.Conv2d(24, bottleneck_width, 1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(bottleneck_width)
        self.bottleneck_relu = nn.ReLU()

        self.expand_conv = nn.Conv2d(bottleneck_width, 24, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(24)
        self.expand_relu = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(24, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
        x = self.block_a(x)
        x = self.down_relu(self.down_bn(self.down_conv(x)))
        x = self.block_b(x)
        x = self.bottleneck_relu(self.bottleneck_bn(self.bottleneck_conv(x)))
        x = self.expand_relu(self.expand_bn(self.expand_conv(x)))
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(x)
