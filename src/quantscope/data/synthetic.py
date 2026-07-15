"""Deterministic synthetic classification data (no downloads, no network).

Each class is a distinct spatial frequency/orientation pattern plus noise,
so a tiny CNN can genuinely learn the task — accuracy is a *measured*
value on real (if synthetic) data, not a fabricated number.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from quantscope.config import DataConfig, ModelConfig

__all__ = ["build_datasets", "make_synthetic_dataset"]


def make_synthetic_dataset(
    *,
    num_samples: int,
    image_size: int,
    num_classes: int,
    in_channels: int,
    seed: int,
    noise_scale: float = 0.35,
) -> TensorDataset:
    """Generate a class-separable synthetic image dataset.

    Class ``k`` is a 2-D sinusoid with class-specific frequency and
    orientation; Gaussian noise keeps the task non-trivial.
    """
    if num_samples < num_classes:
        raise ValueError(
            f"num_samples ({num_samples}) must be >= num_classes ({num_classes})"
        )
    rng = np.random.default_rng(seed)
    ys = np.arange(num_samples) % num_classes
    rng.shuffle(ys)

    coords = np.linspace(0.0, 1.0, image_size)
    xx, yy = np.meshgrid(coords, coords)
    images = np.empty((num_samples, in_channels, image_size, image_size), dtype=np.float32)
    for i, label in enumerate(ys):
        freq = 2.0 + 2.0 * label
        angle = np.pi * label / max(num_classes, 1)
        pattern = np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)))
        noise = rng.normal(scale=noise_scale, size=(in_channels, image_size, image_size))
        images[i] = pattern[None, :, :] + noise

    return TensorDataset(
        torch.from_numpy(images), torch.from_numpy(ys.astype(np.int64))
    )


def build_datasets(
    data_config: DataConfig, model_config: ModelConfig
) -> tuple[TensorDataset, TensorDataset]:
    """Build (train, eval) datasets from configuration."""
    if data_config.name == "texture10":
        from quantscope.data.texture10 import Texture10Params, make_texture10

        params = Texture10Params(
            num_classes=model_config.num_classes,
            image_size=data_config.image_size,
            boundary_fraction=data_config.boundary_fraction,
            boundary_low=data_config.boundary_low,
            boundary_high=data_config.boundary_high,
            snr_db=data_config.snr_db,
            freq_step=data_config.freq_step,
        )
        train = make_texture10(
            num_samples=data_config.num_train, seed=data_config.seed, params=params
        )
        evaluation = make_texture10(
            num_samples=data_config.num_eval, seed=data_config.seed + 1, params=params
        )
        return train, evaluation
    if data_config.name != "synthetic":
        raise ValueError(f"unknown dataset: {data_config.name!r}")
    common = {
        "image_size": data_config.image_size,
        "num_classes": model_config.num_classes,
        "in_channels": model_config.in_channels,
    }
    train = make_synthetic_dataset(
        num_samples=data_config.num_train, seed=data_config.seed, **common
    )
    # Different seed stream for eval so the split is disjoint by construction.
    evaluation = make_synthetic_dataset(
        num_samples=data_config.num_eval, seed=data_config.seed + 1, **common
    )
    return train, evaluation
