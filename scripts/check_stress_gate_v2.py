#!/usr/bin/env python
"""Impulse Stress Gate v2 (ADR-012 addendum 2), fresh dev seed 8, 6-sigma.

Structural early-site definition (fixed before running): the input plus
the activation observers before the first spatial downsampling boundary
(down_conv, stride 2) of BottleneckResNet:
    __input__, stem_relu, block_a.relu1, block_a.relu_out

Criteria:
  1. Early-site reach: >=3 of those 4 sites show MinMax scale increase
     >= 25%, AND the input increases >= 2x.
  2. Behavioral: stressed-calib -> clean-eval MinMax W4A4 NLL worse by
     > 0.02 vs clean-calib -> clean-eval. (Accuracy / prediction flips
     reported, NOT gated.)
  3. Pairing integrity: labels + non-impulse content exactly equal.
  4. No observer shopping: MinMax only; robust observers stay hidden.

Exit 0 = pass (proceed to validation seeds); 1 = fail (close the
impulse mechanism; glints become a separately preregistered family).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from quantscope.benchmark import benchmark_config, texture10_calibration
from quantscope.data.synthetic import build_datasets
from quantscope.data.texture10 import apply_impulse_stress
from quantscope.evaluation.loop import evaluate_detailed
from quantscope.models.tiny_cnn import build_model
from quantscope.quantization.simulate import (
    SimQuantConfig,
    calibrate_activation_params,
    simulate_quantized,
)
from quantscope.sensitivity import predictions

DEV_SEED = 8
RUNS_DIR = "runs/gen-dev8"
STRESS_SEED = 1008
IMPULSE_FRACTION = 0.002
MAGNITUDE = 6.0
EARLY_SITES = ("__input__", "stem_relu", "block_a.relu1", "block_a.relu_out")


def main() -> int:
    config = benchmark_config(seed=DEV_SEED, output_dir=RUNS_DIR, freq_step=0.12)
    checkpoint = Path(RUNS_DIR) / f"texture-a-seed{DEV_SEED}-fp32" / "model.pt"
    model = build_model(config.model)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    clean_calib = texture10_calibration(config)
    stressed_calib = apply_impulse_stress(
        clean_calib, fraction=IMPULSE_FRACTION, magnitude=MAGNITUDE, seed=STRESS_SEED
    )
    _, clean_test = build_datasets(config.data, config.model)

    # Criterion 3: pairing.
    assert torch.equal(clean_calib.tensors[1], stressed_calib.tensors[1])
    changed = float((clean_calib.tensors[0] != stressed_calib.tensors[0]).float().mean())
    print(f"pairing: labels identical; changed-pixel fraction {changed:.5f}")

    # Criterion 1: early-site reach (MinMax only — criterion 4).
    clean_params = calibrate_activation_params(model, clean_calib, bits=4)
    stress_params = calibrate_activation_params(model, stressed_calib, bits=4)
    print(f"\n{'site':<22}{'clean':<12}{'stressed':<12}{'ratio':<8}eligible")
    ratios: dict[str, float] = {}
    for site in clean_params:
        c = float(np.asarray(clean_params[site].scale))
        s = float(np.asarray(stress_params[site].scale))
        ratios[site] = s / c
        print(f"{site:<22}{c:<12.5f}{s:<12.5f}{ratios[site]:<8.3f}{site in EARLY_SITES}")
    early_expanded = sum(ratios[s] >= 1.25 for s in EARLY_SITES)
    input_ratio = ratios["__input__"]
    print(
        f"early sites >=1.25x: {early_expanded}/4 (need >=3); input {input_ratio:.2f}x (need >=2)"
    )

    # Criterion 2: behavioral discrimination on the unchanged clean eval set.
    results = {}
    for label, calib in (("clean_calib", clean_calib), ("stressed_calib", stressed_calib)):
        sim = simulate_quantized(model, calib, SimQuantConfig(4, 4))
        results[label] = {
            "detailed": evaluate_detailed(sim, clean_test),
            "preds": predictions(sim, clean_test),
        }
    d_clean = results["clean_calib"]["detailed"]
    d_stress = results["stressed_calib"]["detailed"]
    degradation = d_stress["nll"] - d_clean["nll"]
    flips = float(np.mean(results["clean_calib"]["preds"] != results["stressed_calib"]["preds"]))
    for label, d in (("clean-calib", d_clean), ("stress-calib", d_stress)):
        print(f"MinMax W4A4 clean-eval, {label}: NLL {d['nll']:.4f} acc {d['accuracy']:.4f}")
    print(
        f"NLL degradation {degradation:+.4f} (gate > 0.02); "
        f"accuracy delta {d_stress['accuracy'] - d_clean['accuracy']:+.4f} (reported); "
        f"prediction flips {flips:.4f} (reported)"
    )

    failures = []
    if early_expanded < 3:
        failures.append(f"1: early-site reach {early_expanded}/4 < 3")
    if input_ratio < 2.0:
        failures.append(f"1: input expansion {input_ratio:.2f}x < 2x")
    if degradation <= 0.02:
        failures.append(f"2: degradation {degradation:+.4f} <= 0.02")
    if failures:
        print("\nGATE v2 FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nGATE v2 PASSED: proceed to validation seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
