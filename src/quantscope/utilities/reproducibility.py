"""Seeding and environment capture for reproducible experiments."""

from __future__ import annotations

import logging
import platform
import random
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

__all__ = ["capture_environment", "set_seed"]

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES = ("torch", "torchvision", "numpy", "quantscope")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for deterministic CPU runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def capture_environment() -> dict[str, object]:
    """Capture platform, interpreter, package, and device information."""
    packages: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "device": "cpu",  # this project is CPU-only by design
        "quantized_engines": list(torch.backends.quantized.supported_engines),
        "git_commit": _git_commit(),
    }
