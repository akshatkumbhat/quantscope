#!/usr/bin/env python
"""ADR-016 Part B: external, direction-only replication of the D-study
primary finding on FashionMNIST.

Claim under test: percentile calibration protects W4A4 NLL against
impulse-contaminated calibration relative to MinMax (stressed
calibration -> clean evaluation). Success = the direction reproduces
on 2 of 2 seeds; magnitudes are reported but are not the claim. A
failed replication is recorded with equal prominence.

Data: FashionMNIST via torchvision (~30 MB one-time download — the
project's only dataset download, documented in ADR-016; cached under
the gitignored /data directory; script-only, never in tests or CI).
Artifacts are standalone provenance-labeled JSON (RunWriter requires
an ExperimentConfig, which does not describe this external dataset;
the deviation is recorded here).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from quantscope.data.texture10 import apply_impulse_stress
from quantscope.evaluation.loop import evaluate_detailed
from quantscope.models.tiny_cnn import TinyCNN
from quantscope.observers import MinMaxObserver, PercentileClippingObserver
from quantscope.quantization.simulate import (
    calibrate_activation_params,
    quantize_weights_uniform,
    simulate_quantized_with_params,
)
from quantscope.regression import atomic_write_json

SEEDS = (0, 1)
OUT_DIR = Path("runs/replication-fashionmnist")
SUMMARY_PATH = OUT_DIR / "replication-summary.json"
NUM_TRAIN, NUM_EVAL, NUM_CALIB = 12_000, 2_000, 256
EPOCHS = 6
STRESS = {"fraction": 0.002, "magnitude": 7.0}


def _tensor_dataset(source, start: int, count: int) -> TensorDataset:
    images = torch.stack([source[i][0] for i in range(start, start + count)])
    labels = torch.tensor([source[i][1] for i in range(start, start + count)])
    return TensorDataset(images, labels)


def _train(model: nn.Module, dataset: TensorDataset, seed: int) -> list[float]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    losses = []
    model.train()
    for _ in range(EPOCHS):
        total, count = 0.0, 0
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        losses.append(total / count)
    model.eval()
    return losses


def main() -> int:
    if SUMMARY_PATH.exists():
        print(f"REFUSED: {SUMMARY_PATH} exists — the replication runs once.")
        return 2
    transform = transforms.ToTensor()
    train_source = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
    test_source = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
    eval_set = _tensor_dataset(test_source, 0, NUM_EVAL)
    calib = _tensor_dataset(train_source, NUM_TRAIN, NUM_CALIB)

    per_seed = {}
    for seed in SEEDS:
        start = time.perf_counter()
        torch.manual_seed(seed)
        model = TinyCNN(num_classes=10, in_channels=1)
        train_set = _tensor_dataset(train_source, 0, NUM_TRAIN)
        losses = _train(model, train_set, seed)
        fp32 = evaluate_detailed(model, eval_set)

        stressed_calib = apply_impulse_stress(calib, seed=1000 + seed, **STRESS)
        wq = quantize_weights_uniform(model, bits=4)
        arms = {}
        for obs_name, factory in (
            ("minmax", MinMaxObserver),
            ("percentile", PercentileClippingObserver),
        ):
            for calib_name, calib_ds in (("clean", calib), ("stressed", stressed_calib)):
                params = calibrate_activation_params(wq, calib_ds, bits=4, observer_factory=factory)
                detail = evaluate_detailed(
                    simulate_quantized_with_params(model, params, weight_bits=4), eval_set
                )
                arms[f"{obs_name}|{calib_name}->clean"] = {
                    "nll": detail["nll"],
                    "accuracy": detail["accuracy"],
                    "provenance": "simulated",
                }
        primary_delta = (
            arms["percentile|stressed->clean"]["nll"] - arms["minmax|stressed->clean"]["nll"]
        )
        per_seed[f"seed{seed}"] = {
            "fp32_eval": {**fp32, "provenance": "measured"},
            "train_losses": losses,
            "arms_w4a4": arms,
            "percentile_minus_minmax_nll_stressed": primary_delta,
            "direction_replicated": primary_delta < 0,
            "wallclock_seconds_measured_dev_cpu": time.perf_counter() - start,
        }
        print(
            f"seed{seed}: FP32 acc {fp32['accuracy']:.4f} | stressed->clean W4A4 NLL "
            f"minmax {arms['minmax|stressed->clean']['nll']:.4f} vs percentile "
            f"{arms['percentile|stressed->clean']['nll']:.4f} (delta {primary_delta:+.4f}) "
            f"-> {'replicated' if primary_delta < 0 else 'NOT replicated'}"
        )

    replicated = all(per_seed[f"seed{s}"]["direction_replicated"] for s in SEEDS)
    summary = {
        "design": "ADR-016 Part B: direction-only replication of D primary finding",
        "dataset": "FashionMNIST (torchvision; 12k train / 2k eval / 256 calib subsets)",
        "stress": {**STRESS, "seed_rule": "1000+seed"},
        "provenance": "task metrics simulated (policy v1) except measured FP32; "
        "direction is the claim, magnitudes reported only",
        "per_seed": per_seed,
        "direction_replicated_on_all_seeds": replicated,
        "verdict": "REPLICATED" if replicated else "NOT REPLICATED",
    }
    atomic_write_json(SUMMARY_PATH, summary)
    print(f"\nsummary: {SUMMARY_PATH}\nVERDICT: {summary['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
