"""Texture-10: a margin-controlled synthetic texture classification benchmark.

Each class is a prototype of two overlapping sinusoidal components with
small inter-class parameter separations. Difficulty is controlled by:

- **boundary examples**: a fraction of samples are interpolated 30-45% of
  the way toward a neighboring class prototype (label preserved), placing
  them near a real decision boundary;
- **SNR**: white + low-frequency noise scaled to a target signal-to-noise
  ratio;
- nuisance jitter: global rotation, per-component frequency jitter, random
  phase (equivalent to translation for sinusoids), contrast, mild blur.

Rotation and translation are applied analytically (rotating orientations,
shifting phases) rather than by image warping, so there are no
interpolation artifacts. Deterministic given a seed. No downloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import TensorDataset

__all__ = ["Texture10Params", "apply_impulse_stress", "make_texture10", "texture10_splits"]


@dataclass(frozen=True)
class Texture10Params:
    """Generation parameters (defaults follow the benchmark design)."""

    num_classes: int = 10
    image_size: int = 32
    boundary_fraction: float = 0.2  # fraction of samples near a class boundary
    boundary_low: float = 0.30  # interpolation range toward the neighbor
    boundary_high: float = 0.45
    snr_db: float = 8.0
    rotation_deg: float = 7.0
    freq_jitter: float = 0.04
    contrast_low: float = 0.75
    contrast_high: float = 1.25
    blur_sigma_max: float = 0.8
    # Inter-class frequency separation of the primary component. Smaller
    # steps shrink class margins globally. This knob exists to reduce class
    # margin, NOT to manufacture a particular layer sensitivity ranking —
    # any ranking that is meaningful, heterogeneous, and stable is valid
    # (ADR-009).
    freq_step: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.boundary_fraction <= 0.5:
            raise ValueError("boundary_fraction must be in [0, 0.5]")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if self.freq_step <= 0.0:
            raise ValueError("freq_step must be positive")


def _class_components(
    k: int, num_classes: int, freq_step: float
) -> list[tuple[float, float, float]]:
    """(frequency, orientation, amplitude) for class ``k``'s two components.

    Adjacent classes differ by one small orientation step and a small
    frequency increment, keeping inter-class margins deliberately tight.
    """
    step = np.pi / num_classes
    return [
        (3.0 + freq_step * k, k * step, 1.0),
        (6.5, k * step + np.deg2rad(55.0), 0.55),
    ]


def _render_pattern(
    components: list[tuple[float, float, float]],
    xx: np.ndarray,
    yy: np.ndarray,
    rng: np.random.Generator,
    params: Texture10Params,
    rotation: float,
) -> np.ndarray:
    """Render one jittered class pattern (no noise)."""
    out = np.zeros_like(xx)
    for freq, theta, amp in components:
        f = freq * (1.0 + rng.uniform(-params.freq_jitter, params.freq_jitter))
        angle = theta + rotation
        phase = rng.uniform(0.0, 2.0 * np.pi)  # random phase == translation
        out += amp * np.sin(
            2.0 * np.pi * f * (xx * np.cos(angle) + yy * np.sin(angle)) + phase
        )
    return out


def make_texture10(
    *,
    num_samples: int,
    seed: int,
    params: Texture10Params | None = None,
) -> TensorDataset:
    """Generate a Texture-10 dataset (images float32, labels int64)."""
    p = params or Texture10Params()
    if num_samples < p.num_classes:
        raise ValueError(
            f"num_samples ({num_samples}) must be >= num_classes ({p.num_classes})"
        )
    rng = np.random.default_rng(seed)
    size = p.image_size
    coords = np.linspace(0.0, 1.0, size, endpoint=False)
    xx, yy = np.meshgrid(coords, coords)

    labels = np.arange(num_samples) % p.num_classes
    rng.shuffle(labels)
    n_boundary = round(num_samples * p.boundary_fraction)
    boundary_mask = np.zeros(num_samples, dtype=bool)
    boundary_mask[rng.choice(num_samples, size=n_boundary, replace=False)] = True

    images = np.empty((num_samples, 1, size, size), dtype=np.float32)
    for i, label in enumerate(labels):
        rotation = np.deg2rad(rng.uniform(-p.rotation_deg, p.rotation_deg))
        pattern = _render_pattern(
            _class_components(int(label), p.num_classes, p.freq_step), xx, yy, rng, p, rotation
        )
        if boundary_mask[i]:
            neighbor = (int(label) + rng.choice([-1, 1])) % p.num_classes
            lam = rng.uniform(p.boundary_low, p.boundary_high)
            neighbor_pattern = _render_pattern(
                _class_components(neighbor, p.num_classes, p.freq_step), xx, yy, rng, p, rotation
            )
            pattern = (1.0 - lam) * pattern + lam * neighbor_pattern

        pattern *= rng.uniform(p.contrast_low, p.contrast_high)

        # Noise scaled to the per-sample target SNR: white + low-frequency.
        signal_power = float(np.mean(pattern**2))
        noise_power = signal_power / (10.0 ** (p.snr_db / 10.0))
        white = rng.normal(size=(size, size))
        low_freq = gaussian_filter(rng.normal(size=(size, size)), sigma=3.0)
        low_freq /= max(float(low_freq.std()), 1e-8)
        noise = 0.7 * white + 0.3 * low_freq
        noise *= np.sqrt(noise_power) / max(float(noise.std()), 1e-8)
        img = pattern + noise

        sigma = rng.uniform(0.0, p.blur_sigma_max)
        if sigma > 0.05:
            img = gaussian_filter(img, sigma=sigma)
        images[i, 0] = img.astype(np.float32)

    return TensorDataset(
        torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64))
    )


def apply_impulse_stress(
    dataset: TensorDataset,
    *,
    fraction: float = 0.002,
    magnitude: float = 6.0,
    seed: int = 0,
) -> TensorDataset:
    """Impulse nuisance outliers for the ADR-012 stress condition.

    Operates on a FINISHED clean dataset (post-blur, since generation
    already applied blur), so labels, sample IDs, and every texture
    parameter are preserved by construction — the returned dataset is a
    paired copy of the input.

    Per image: ``fraction`` of pixels are replaced by ``+/-magnitude``
    times that image's standard deviation, signs balanced
    deterministically (alternating +/- over the selected pixels).
    Label-independent by construction.
    """
    if not 0.0 < fraction <= 0.05:
        raise ValueError("fraction must be in (0, 0.05]")
    if magnitude <= 0:
        raise ValueError("magnitude must be positive")
    images, labels = dataset.tensors
    stressed = images.clone().numpy()
    rng = np.random.default_rng(seed)
    num_pixels = stressed.shape[-2] * stressed.shape[-1]
    num_impulses = max(1, round(fraction * num_pixels))
    for i in range(stressed.shape[0]):
        std = float(stressed[i].std())
        flat = stressed[i].reshape(-1)
        idx = rng.choice(flat.size, size=num_impulses, replace=False)
        signs = np.where(np.arange(num_impulses) % 2 == 0, 1.0, -1.0)
        flat[idx] = signs * magnitude * std
    return TensorDataset(torch.from_numpy(stressed), labels.clone())


def texture10_splits(
    *,
    num_train: int = 5000,
    num_test: int = 2000,
    num_calib: int = 256,
    seed: int = 0,
    params: Texture10Params | None = None,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Build disjoint (train, test, calibration) splits by seed stream."""
    return (
        make_texture10(num_samples=num_train, seed=seed, params=params),
        make_texture10(num_samples=num_test, seed=seed + 1, params=params),
        make_texture10(num_samples=num_calib, seed=seed + 2, params=params),
    )
