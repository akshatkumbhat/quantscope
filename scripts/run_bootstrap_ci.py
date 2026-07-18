#!/usr/bin/env python
"""ADR-016 Part C: bootstrap 95% CIs for the headline paired deltas.

Nonparametric bootstrap over evaluation samples (n=2000, B=10,000,
numpy seed 0, percentile intervals) for the QAT study (QAT-PTQ,
QAT-control, control-PTQ) and the D-study primary condition
(percentile-MinMax, MSE-grid-MinMax), per validation seed.

Evaluation-only recomputation from frozen checkpoints and saved
fine-tuned weights through the identical pipelines; every recomputed
mean NLL must match its recorded artifact aggregate within 1e-6 or the
run stops as a provenance failure. CIs supplement, never replace, the
recorded point values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from quantscope.benchmark import benchmark_config, texture10_calibration
from quantscope.data.synthetic import build_datasets
from quantscope.data.texture10 import apply_impulse_stress
from quantscope.models.tiny_cnn import build_model
from quantscope.observers import MinMaxObserver, MSEGridSearchObserver, PercentileClippingObserver
from quantscope.quantization.simulate import (
    SimQuantConfig,
    calibrate_activation_params,
    quantize_weights_uniform,
    simulate_quantized,
    simulate_quantized_with_params,
)
from quantscope.regression import atomic_write_json

SEEDS = (0, 1, 2)
VALIDATION_DIR = Path("runs/validation-012")
SUMMARY_PATH = VALIDATION_DIR / "bootstrap-ci-summary.json"
B = 10_000
TOL = 1e-6


def _per_sample_nll(model: nn.Module, eval_set) -> np.ndarray:
    loader = DataLoader(eval_set, batch_size=64, shuffle=False)
    out = []
    with torch.no_grad():
        for images, labels in loader:
            out.append(nn.functional.cross_entropy(model(images), labels, reduction="none").numpy())
    return np.concatenate(out).astype(np.float64)


def _consistency(name: str, recomputed_mean: float, recorded: float) -> None:
    if abs(recomputed_mean - recorded) > TOL:
        raise RuntimeError(
            f"provenance failure: {name} recomputed mean {recomputed_mean!r} vs recorded "
            f"{recorded!r} (tolerance {TOL})"
        )


def _ci(delta: np.ndarray, idx: np.ndarray) -> dict:
    means = delta[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "point": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def main() -> int:
    if SUMMARY_PATH.exists():
        print(f"REFUSED: {SUMMARY_PATH} exists — the bootstrap runs once.")
        return 2
    qat_summary = json.loads((VALIDATION_DIR / "qat-study-summary.json").read_text())["per_seed"]
    control_summary = json.loads((VALIDATION_DIR / "qat-control-summary.json").read_text())[
        "per_seed"
    ]

    rng = np.random.default_rng(0)
    results: dict = {}
    for seed in SEEDS:
        config = benchmark_config(seed=seed, output_dir=str(VALIDATION_DIR), freq_step=0.12)
        model = build_model(config.model)
        model.load_state_dict(
            torch.load(VALIDATION_DIR / f"texture-a-seed{seed}-fp32" / "model.pt")
        )
        model.eval()
        _, eval_set = build_datasets(config.data, config.model)
        calib = texture10_calibration(config)

        # --- QAT study arms (identical pipelines to the recorded runs).
        act_params = calibrate_activation_params(
            quantize_weights_uniform(model, bits=4), calib, bits=4
        )
        ptq = _per_sample_nll(
            simulate_quantized_with_params(model, act_params, weight_bits=4), eval_set
        )
        qat_model = build_model(config.model)
        qat_model.load_state_dict(
            torch.load(VALIDATION_DIR / f"texture-a-seed{seed}-qat-w4a4" / "model_qat_fp32.pt")
        )
        qat_model.eval()
        qat = _per_sample_nll(
            simulate_quantized_with_params(qat_model, act_params, weight_bits=4), eval_set
        )
        control_model = build_model(config.model)
        control_model.load_state_dict(
            torch.load(
                VALIDATION_DIR / f"texture-a-seed{seed}-qat-control" / "model_control_fp32.pt"
            )
        )
        control_model.eval()
        control_ptq = _per_sample_nll(
            simulate_quantized(control_model, calib, SimQuantConfig(4, 4)), eval_set
        )
        recorded = qat_summary[f"seed{seed}"]
        _consistency(f"seed{seed} ptq", ptq.mean(), recorded["ptq_nll"])
        _consistency(f"seed{seed} qat", qat.mean(), recorded["qat_nll"])
        _consistency(
            f"seed{seed} control_ptq",
            control_ptq.mean(),
            control_summary[f"seed{seed}"]["control_ptq_nll"],
        )

        # --- D-study primary condition arms (stressed calib -> clean eval).
        stressed_calib = apply_impulse_stress(
            calib, fraction=0.002, magnitude=7.0, seed=1000 + seed
        )
        d_metrics = json.loads(
            (VALIDATION_DIR / f"texture-a-seed{seed}-observer-study" / "metrics.json").read_text()
        )["metrics"]
        d_by_name = {m["name"]: m["value"] for m in d_metrics}
        wq = quantize_weights_uniform(model, bits=4)
        d_arms = {}
        for obs_name, factory in (
            ("minmax", MinMaxObserver),
            ("percentile", PercentileClippingObserver),
            ("mse_grid", MSEGridSearchObserver),
        ):
            params = calibrate_activation_params(
                wq, stressed_calib, bits=4, observer_factory=factory
            )
            nlls = _per_sample_nll(
                simulate_quantized_with_params(model, params, weight_bits=4), eval_set
            )
            _consistency(
                f"seed{seed} D {obs_name}",
                nlls.mean(),
                d_by_name[f"{obs_name}|W4A4|stressed->clean"]["nll"],
            )
            d_arms[obs_name] = nlls

        # --- Paired bootstrap: one shared index matrix per seed.
        idx = rng.integers(0, len(eval_set), size=(B, len(eval_set)))
        results[f"seed{seed}"] = {
            "qat_minus_ptq_nll": _ci(qat - ptq, idx),
            "qat_minus_control_ptq_nll": _ci(qat - control_ptq, idx),
            "control_ptq_minus_ptq_nll": _ci(control_ptq - ptq, idx),
            "percentile_minus_minmax_nll_stressed": _ci(
                d_arms["percentile"] - d_arms["minmax"], idx
            ),
            "mse_grid_minus_minmax_nll_stressed": _ci(d_arms["mse_grid"] - d_arms["minmax"], idx),
        }
        print(f"seed{seed}: " + json.dumps(results[f"seed{seed}"], indent=1))

    summary = {
        "design": "ADR-016 Part C: paired nonparametric bootstrap over eval samples",
        "provenance": "simulated NLLs recomputed from frozen models (consistency-gated 1e-6); "
        "CIs supplement recorded point values",
        "n_samples": 2000,
        "n_resamples": B,
        "rng_seed": 0,
        "per_seed": results,
    }
    atomic_write_json(SUMMARY_PATH, summary)
    print(f"summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
