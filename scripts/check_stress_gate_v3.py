#!/usr/bin/env python
"""Impulse Stress Gate v3 (ADR-012 addendum 4), fresh dev seed 6, 7-sigma.

Gates v1 and v2 remain FAILED as originally preregistered. v3 is a new
prospective mechanism variant whose SOLE design change is impulse
magnitude 6-sigma -> 7-sigma; the 2.0x input-expansion threshold is
retained. The rationale is arithmetic consistency with measured
clean-input extrema (~3.06 sigma on the v2 calibration split), not
optimization against observer performance.

Criteria (identical to Gate v2, ADR-012 addendum 2):
  1. Early-site reach: >=3 of {__input__, stem_relu, block_a.relu1,
     block_a.relu_out} show MinMax scale increase >= 25%, AND the input
     increases >= 2x.
  2. Behavioral: stressed-calib -> clean-eval MinMax W4A4 NLL worse by
     > 0.02 vs clean-calib -> clean-eval. (Accuracy / prediction flips
     reported, NOT gated.)
  3. Pairing integrity: labels + non-impulse content exactly equal.
  4. No observer shopping: MinMax only; robust observers stay hidden.

ONE attempt only: the script refuses to run if the gate artifact
already exists. No fallback magnitude; no threshold adjustment after
seeing results.

Exit 0 = pass (proceed to validation seeds); 1 = fail (close the
impulse stress family for this phase; stop and report).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from quantscope.analysis.stress_gate import GATE_V3_SPEC, GATE_V3_STRESS, evaluate_stress_gate
from quantscope.benchmark import benchmark_config, texture10_calibration
from quantscope.config import Provenance
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
from quantscope.utilities import RunWriter

DEV_SEED = 6
RUNS_DIR = "runs/gen-dev6"
SIM_NOTE = "fake-quant simulation policy v1; not integer execution"


def main() -> int:
    config = benchmark_config(seed=DEV_SEED, output_dir=RUNS_DIR, freq_step=0.12)
    artifact_dir = Path(RUNS_DIR) / f"{config.run_name}-stress-gate-v3"
    if artifact_dir.exists():
        print(f"REFUSED: {artifact_dir} already exists — Gate v3 allows exactly one attempt.")
        return 2

    checkpoint = Path(RUNS_DIR) / f"texture-a-seed{DEV_SEED}-fp32" / "model.pt"
    model = build_model(config.model)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    clean_calib = texture10_calibration(config)
    stressed_calib = apply_impulse_stress(
        clean_calib,
        fraction=GATE_V3_STRESS.fraction,
        magnitude=GATE_V3_STRESS.magnitude,
        seed=GATE_V3_STRESS.seed,
    )
    _, clean_test = build_datasets(config.data, config.model)

    # Criterion 3: pairing.
    pairing_ok = torch.equal(clean_calib.tensors[1], stressed_calib.tensors[1])
    changed = float((clean_calib.tensors[0] != stressed_calib.tensors[0]).float().mean())
    print(f"pairing: labels identical={pairing_ok}; changed-pixel fraction {changed:.5f}")

    # Criterion 1: early-site reach (MinMax only — criterion 4).
    clean_params = calibrate_activation_params(model, clean_calib, bits=4)
    stress_params = calibrate_activation_params(model, stressed_calib, bits=4)
    print(f"\n{'site':<22}{'clean':<12}{'stressed':<12}{'ratio':<8}eligible")
    ratios: dict[str, float] = {}
    for site in clean_params:
        c = float(np.asarray(clean_params[site].scale))
        s = float(np.asarray(stress_params[site].scale))
        ratios[site] = s / c
        eligible = site in GATE_V3_SPEC.early_sites
        print(f"{site:<22}{c:<12.5f}{s:<12.5f}{ratios[site]:<8.3f}{eligible}")

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
    accuracy_delta = d_stress["accuracy"] - d_clean["accuracy"]
    flips = float(np.mean(results["clean_calib"]["preds"] != results["stressed_calib"]["preds"]))
    for label, d in (("clean-calib", d_clean), ("stress-calib", d_stress)):
        print(f"MinMax W4A4 clean-eval, {label}: NLL {d['nll']:.4f} acc {d['accuracy']:.4f}")
    print(
        f"NLL degradation {degradation:+.4f} (gate > 0.02); "
        f"accuracy delta {accuracy_delta:+.4f} (reported); "
        f"prediction flips {flips:.4f} (reported)"
    )

    verdict = evaluate_stress_gate(GATE_V3_SPEC, ratios, degradation, pairing_ok)
    print(
        f"early sites >=1.25x: {verdict.early_expanded}/{len(GATE_V3_SPEC.early_sites)} "
        f"(need >={GATE_V3_SPEC.min_early_expanded}); "
        f"input {verdict.input_ratio:.2f}x (need >={GATE_V3_SPEC.input_expansion_threshold}x)"
    )

    writer = RunWriter(config, kind="stress-gate-v3")
    writer.record_metric(
        "stress_spec",
        {
            "fraction": GATE_V3_STRESS.fraction,
            "magnitude_sigma": GATE_V3_STRESS.magnitude,
            "stress_seed": GATE_V3_STRESS.seed,
        },
        Provenance.MEASURED,
        note="preregistered intervention parameters (ADR-012 addendum 4)",
    )
    for site, ratio in ratios.items():
        writer.record_metric(
            f"minmax_scale_ratio[{site}]",
            ratio,
            Provenance.MEASURED,
            note="stressed/clean MinMax observer scale on the calibration split",
        )
    writer.record_metric("pairing_labels_identical", int(pairing_ok), Provenance.MEASURED)
    writer.record_metric("changed_pixel_fraction", changed, Provenance.MEASURED)
    for name, value in (
        ("w4a4_nll_clean_calib", d_clean["nll"]),
        ("w4a4_nll_stressed_calib", d_stress["nll"]),
        ("w4a4_nll_degradation", degradation),
        ("w4a4_accuracy_delta", accuracy_delta),
        ("w4a4_prediction_flips", flips),
    ):
        writer.record_metric(name, value, Provenance.SIMULATED, note=SIM_NOTE)
    writer.record_metric(
        "gate_passed",
        int(verdict.passed),
        Provenance.MEASURED,
        failures=list(verdict.failures),
        note="Gate v3 verdict under the preregistered ADR-012 addendum 4 criteria",
    )
    print(f"artifact: {writer.finalize()}")

    if not verdict.passed:
        print("\nGATE v3 FAILED:")
        for failure in verdict.failures:
            print(f"  - {failure}")
        return 1
    print("\nGATE v3 PASSED: proceed to validation seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
