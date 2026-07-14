#!/usr/bin/env python
"""ADR-012 stress-design gate, dev seed 7 (run BEFORE validation seeds).

Requirements (amended, 2026-07-14):
  a. Stressed calibration expands the MinMax scale by >= 25% at >= 5 of
     the 9 policy-v1 activation sites (the "7 of 14" amendment adapted
     proportionally to policy v1's site count, recorded in ADR-012).
  b. MinMax W4A4 NLL on the UNCHANGED clean evaluation set worsens by
     > 0.02 under stressed vs clean calibration.
  c. Labels and non-impulse pixels identical between paired sets.

Fallback: rerun with --magnitude 10 (decided ONLY on these metrics).
Exit 0 = gate passed at this magnitude; 1 = failed.
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

DEV_SEED = 7
STRESS_SEED = 1007  # fixed; independent of generator seed streams
IMPULSE_FRACTION = 0.002


def main(magnitude: float = 6.0) -> int:
    config = benchmark_config(seed=DEV_SEED, output_dir="runs/gen-c3", freq_step=0.12)
    checkpoint = Path("runs/gen-c3") / f"texture-a-seed{DEV_SEED}-fp32" / "model.pt"
    model = build_model(config.model)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    clean_calib = texture10_calibration(config)
    stressed_calib = apply_impulse_stress(
        clean_calib, fraction=IMPULSE_FRACTION, magnitude=magnitude, seed=STRESS_SEED
    )
    _, clean_test = build_datasets(config.data, config.model)

    # (c) pairing checks.
    assert torch.equal(clean_calib.tensors[1], stressed_calib.tensors[1])
    changed = (clean_calib.tensors[0] != stressed_calib.tensors[0]).float().mean()
    print(f"paired: labels identical; fraction of changed pixels {float(changed):.5f}")

    # (a) MinMax scale expansion per site at activation bits = 4.
    clean_params = calibrate_activation_params(model, clean_calib, bits=4)
    stress_params = calibrate_activation_params(model, stressed_calib, bits=4)
    print(f"\n{'site':<22}{'clean scale':<14}{'stressed':<14}ratio")
    expanded = 0
    for site in clean_params:
        c = float(np.asarray(clean_params[site].scale))
        s = float(np.asarray(stress_params[site].scale))
        ratio = s / c
        expanded += ratio >= 1.25
        print(f"{site:<22}{c:<14.5f}{s:<14.5f}{ratio:.3f}")
    print(f"sites with >=25% expansion: {expanded}/9 (need >=5)")

    # (b) MinMax W4A4 on the clean test set, clean vs stressed calibration.
    nll = {}
    for label, calib in (("clean_calib", clean_calib), ("stressed_calib", stressed_calib)):
        sim = simulate_quantized(model, calib, SimQuantConfig(4, 4))
        nll[label] = evaluate_detailed(sim, clean_test)["nll"]
        print(f"W4A4 MinMax, {label} -> clean eval: NLL {nll[label]:.4f}")
    degradation = nll["stressed_calib"] - nll["clean_calib"]
    print(f"NLL degradation from stressed calibration: {degradation:+.4f} (need > 0.02)")

    failures = []
    if expanded < 5:
        failures.append(f"a: only {expanded}/9 sites expanded >=25%")
    if degradation <= 0.02:
        failures.append(f"b: degradation {degradation:+.4f} <= 0.02")
    if failures:
        print(f"\nGATE FAILED at magnitude {magnitude}:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nGATE PASSED at magnitude {magnitude}")
    return 0


if __name__ == "__main__":
    magnitude = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    sys.exit(main(magnitude))
