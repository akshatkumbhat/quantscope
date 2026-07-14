#!/usr/bin/env python
"""ADR-012 plan step D: observer-policy comparison study (validation seeds).

Runs only after Impulse Stress Gate v3 PASSED (ADR-012 addendum 5).
Design frozen in ADR-012 (+ addenda 2 and 4); no observer-parameter
tuning is permitted here.

- Checkpoints: the three frozen clean-trained freq_step=0.12 validation
  checkpoints (seeds 0/1/2, runs/validation-012/). No retraining.
- Observer arms (activations only; weights are always per-channel
  symmetric MinMax): MinMax baseline; Percentile 0.1/99.9; MSE-grid
  frozen defaults; PowerOfTwo round-up.
- Configurations: W4A4 primary; W8A4 activation-isolation; W8A8
  backend-like. Full paired 2x2 factorial (calibration x evaluation,
  clean/stressed) for every configuration; the stressed-evaluation
  arms of W8A4/W8A8 are cheap secondary extras per ADR-012 item 5.
  Primary condition: stressed calibration -> clean evaluation, W4A4.
- Stress: the Gate-v3 impulse intervention (fraction 0.002, 7-sigma),
  applied to finished clean datasets so pairing holds by construction.
  Stress seeds fixed before the run: calibration 1000+seed (v1/v2/v3
  convention), evaluation 2000+seed, probe 3000+seed.
- Mechanism evidence: per-site calibrated scales; per-site SQNR and
  saturation on a held-out probe batch (seed stream +3, 256 samples,
  act bits 4) for both the clean probe and its paired stressed probe;
  exact power-of-two property check for the pow2 arm.
- Mechanism decomposition (MinMax, W4A4, clean eval): stressed qparams
  substituted one site at a time, then cumulatively by stage (input /
  remaining early sites / deeper sites).
- Predeclared interpretation (ADR-012 item 8): Q1 robustness, Q2
  clean-data non-inferiority (0.005 mean-NLL tolerance, per
  configuration), Q3 power-of-two cost (measurement only). Cross-seed
  ranking stability reported, not gated. Negative results get equal
  prominence.

Every persisted metric is provenance-labeled (measured vs simulated).
All quantized-model numbers are fake-quant simulation policy v1, NOT
integer execution, and no number here is NPU/hardware performance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from quantscope.analysis.metrics import saturation_rate, sqnr_db
from quantscope.analysis.stress_gate import GATE_V3_SPEC, GATE_V3_STRESS
from quantscope.benchmark import benchmark_config, texture10_calibration
from quantscope.config import Provenance
from quantscope.data.synthetic import build_datasets
from quantscope.data.texture10 import Texture10Params, apply_impulse_stress, make_texture10
from quantscope.evaluation.loop import evaluate_detailed
from quantscope.models.tiny_cnn import build_model
from quantscope.observers import (
    MinMaxObserver,
    MSEGridSearchObserver,
    PercentileClippingObserver,
    PowerOfTwoScaleObserver,
)
from quantscope.quantization.affine import quantize
from quantscope.quantization.simulate import (
    SimQuantConfig,
    calibrate_activation_params,
    quantize_weights_uniform,
    simulate_quantized_with_params,
)
from quantscope.utilities import RunWriter

SEEDS = (0, 1, 2)
RUNS_DIR = "runs/validation-012"
SUMMARY_PATH = Path(RUNS_DIR) / "observer-study-summary.json"
PROBE_SIZE = 256
SIM_NOTE = "fake-quant simulation policy v1; not integer execution"

OBSERVERS = {
    "minmax": MinMaxObserver,
    "percentile": PercentileClippingObserver,  # frozen 0.1/99.9 defaults
    "mse_grid": MSEGridSearchObserver,  # frozen defaults (32 candidates, 0.3)
    "pow2": PowerOfTwoScaleObserver,  # frozen round-up default
}
ROBUST_OBSERVERS = ("percentile", "mse_grid")  # Q1/Q2 candidates
CONFIGS = (SimQuantConfig(4, 4), SimQuantConfig(8, 4), SimQuantConfig(8, 8))
PRIMARY = "W4A4"
CONDITIONS = (
    ("clean", "clean"),
    ("stressed", "clean"),  # primary
    ("clean", "stressed"),
    ("stressed", "stressed"),
)

# Q1/Q2 thresholds (ADR-012 item 8; predeclared, do not adjust).
Q1_NLL_IMPROVEMENT = 0.01
Q1_MIN_FAVORABLE_SEEDS = 2
Q1_MAX_ACCURACY_LOSS = 0.005  # 0.5 pp
Q2_NLL_TOLERANCE = 0.005

EARLY_SITES = GATE_V3_SPEC.early_sites  # input + pre-downsampling sites


def _make_probe(config):
    """Held-out probe batch, seed stream +3 (never the calibration batch)."""
    return make_texture10(
        num_samples=PROBE_SIZE,
        seed=config.data.seed + 3,
        params=Texture10Params(
            num_classes=config.model.num_classes,
            image_size=config.data.image_size,
            boundary_fraction=config.data.boundary_fraction,
            boundary_low=config.data.boundary_low,
            boundary_high=config.data.boundary_high,
            snr_db=config.data.snr_db,
            freq_step=config.data.freq_step,
        ),
    )


def _stress(dataset, seed: int):
    return apply_impulse_stress(
        dataset,
        fraction=GATE_V3_STRESS.fraction,
        magnitude=GATE_V3_STRESS.magnitude,
        seed=seed,
    )


def _check_pairing(name: str, clean, stressed) -> None:
    if not torch.equal(clean.tensors[1], stressed.tensors[1]):
        raise AssertionError(f"pairing violated for {name}: labels differ")
    changed = (clean.tensors[0] != stressed.tensors[0]).float().mean()
    if not 0.0 < float(changed) < 0.01:
        raise AssertionError(f"pairing violated for {name}: changed fraction {changed}")


def _capture_fp32_activations(model, images: torch.Tensor) -> dict[str, np.ndarray]:
    """FP32 activations at every policy-v1 site (input + ReLUs)."""
    captured: dict[str, np.ndarray] = {"__input__": images.numpy()}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):

            def hook(_m, _i, out, *, _name=name):
                captured[_name] = out.detach().numpy()

            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(images)
    for h in handles:
        h.remove()
    return captured


def _pow2_exact(params) -> bool:
    mantissa, _ = np.frexp(np.asarray(params.scale, dtype=np.float64))
    return bool(np.all(mantissa == 0.5))


def run_seed(seed: int) -> dict:
    config = benchmark_config(seed=seed, output_dir=RUNS_DIR, freq_step=0.12)
    checkpoint = Path(RUNS_DIR) / f"texture-a-seed{seed}-fp32" / "model.pt"
    model = build_model(config.model)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    clean_calib = texture10_calibration(config)
    stressed_calib = _stress(clean_calib, 1000 + seed)
    _, clean_eval = build_datasets(config.data, config.model)
    stressed_eval = _stress(clean_eval, 2000 + seed)
    clean_probe = _make_probe(config)
    stressed_probe = _stress(clean_probe, 3000 + seed)
    _check_pairing("calibration", clean_calib, stressed_calib)
    _check_pairing("evaluation", clean_eval, stressed_eval)
    _check_pairing("probe", clean_probe, stressed_probe)
    calibs = {"clean": clean_calib, "stressed": stressed_calib}
    evals = {"clean": clean_eval, "stressed": stressed_eval}

    writer = RunWriter(config, kind="observer-study")
    writer.record_metric(
        "stress_spec",
        {
            "fraction": GATE_V3_STRESS.fraction,
            "magnitude_sigma": GATE_V3_STRESS.magnitude,
            "calib_stress_seed": 1000 + seed,
            "eval_stress_seed": 2000 + seed,
            "probe_stress_seed": 3000 + seed,
        },
        Provenance.MEASURED,
        note="Gate-v3 impulse intervention (ADR-012 addenda 4-5)",
    )

    fp32: dict[str, dict[str, float]] = {}
    for eval_name, dataset in evals.items():
        fp32[eval_name] = evaluate_detailed(model, dataset)
        for metric in ("nll", "accuracy"):
            writer.record_metric(
                f"fp32[{eval_name}_eval][{metric}]",
                fp32[eval_name][metric],
                Provenance.MEASURED,
            )

    # Policy v1 calibrates activations against the weight-quantized
    # model, so calibration is per (observer, condition, configuration);
    # the recorded scales are exactly those the simulated models use
    # (verified by unit test against simulate_quantized).
    weight_quantized = {
        bits: quantize_weights_uniform(model, bits=bits)
        for bits in sorted({c.weight_bits for c in CONFIGS})
    }
    act_params: dict[tuple[str, str, str], dict] = {}
    for obs_name, factory in OBSERVERS.items():
        for calib_name, calib in calibs.items():
            for cfg in CONFIGS:
                params = calibrate_activation_params(
                    weight_quantized[cfg.weight_bits],
                    calib,
                    bits=cfg.act_bits,
                    observer_factory=factory,
                )
                act_params[(obs_name, calib_name, cfg.label)] = params
                writer.record_metric(
                    f"scales[{obs_name}][{calib_name}][{cfg.label}]",
                    {site: float(np.asarray(p.scale)) for site, p in params.items()},
                    Provenance.MEASURED,
                    note="per-site calibrated activation scale",
                )
                if obs_name == "pow2":
                    exact = all(_pow2_exact(p) for p in params.values())
                    writer.record_metric(
                        f"pow2_scales_exact[{calib_name}][{cfg.label}]",
                        int(exact),
                        Provenance.MEASURED,
                        note="every scale verified an exact power of two",
                    )
                    if not exact:
                        raise AssertionError("pow2 observer produced a non-power-of-two scale")

    # Mechanism evidence: per-site SQNR + saturation on the probe batch
    # under the primary W4A4 configuration. Reference activations come
    # from the W4-weight-quantized model — the tensors the activation
    # quantizers actually see (before propagation effects).
    probe_acts = {
        "clean": _capture_fp32_activations(weight_quantized[4], clean_probe.tensors[0]),
        "stressed": _capture_fp32_activations(weight_quantized[4], stressed_probe.tensors[0]),
    }
    for (obs_name, calib_name, cfg_label), params in act_params.items():
        if cfg_label != PRIMARY:
            continue
        for probe_name, acts in probe_acts.items():
            sqnr = {}
            saturation = {}
            for site, p in params.items():
                reference = acts[site]
                codes = quantize(reference, p)
                dequantized = (codes - p.zero_point).astype(np.float64) * p.scale
                sqnr[site] = sqnr_db(reference, dequantized)
                saturation[site] = saturation_rate(codes, p.qmin, p.qmax)
            writer.record_metric(
                f"probe_sqnr_db[{obs_name}][{calib_name}_calib][{probe_name}_probe]",
                sqnr,
                Provenance.SIMULATED,
                note=SIM_NOTE,
            )
            writer.record_metric(
                f"probe_saturation[{obs_name}][{calib_name}_calib][{probe_name}_probe]",
                saturation,
                Provenance.SIMULATED,
                note=SIM_NOTE,
            )

    # The factorial: observer x configuration x calibration x evaluation.
    results: dict[str, dict] = {}
    for obs_name in OBSERVERS:
        for cfg in CONFIGS:
            sim = {
                calib_name: simulate_quantized_with_params(
                    model,
                    act_params[(obs_name, calib_name, cfg.label)],
                    weight_bits=cfg.weight_bits,
                )
                for calib_name in calibs
            }
            for calib_name, eval_name in CONDITIONS:
                detailed = evaluate_detailed(sim[calib_name], evals[eval_name])
                key = f"{obs_name}|{cfg.label}|{calib_name}->{eval_name}"
                results[key] = {
                    "nll": detailed["nll"],
                    "accuracy": detailed["accuracy"],
                    "delta_nll_vs_fp32": detailed["nll"] - fp32[eval_name]["nll"],
                    "delta_acc_vs_fp32": detailed["accuracy"] - fp32[eval_name]["accuracy"],
                }
                writer.record_metric(key, results[key], Provenance.SIMULATED, note=SIM_NOTE)
                print(
                    f"  seed{seed} {key}: nll={detailed['nll']:.4f} acc={detailed['accuracy']:.4f}"
                )

    # Mechanism decomposition: MinMax W4A4, clean evaluation.
    clean_params = act_params[("minmax", "clean", PRIMARY)]
    stress_params = act_params[("minmax", "stressed", PRIMARY)]
    base_nll = results[f"minmax|{PRIMARY}|clean->clean"]["nll"]
    full_nll = results[f"minmax|{PRIMARY}|stressed->clean"]["nll"]

    def _nll_with(sites: tuple[str, ...]) -> float:
        mixed = {s: (stress_params[s] if s in sites else clean_params[s]) for s in clean_params}
        sim = simulate_quantized_with_params(model, mixed, weight_bits=4)
        return evaluate_detailed(sim, clean_eval)["nll"]

    single_site = {site: _nll_with((site,)) - base_nll for site in clean_params}
    early_rest = tuple(s for s in EARLY_SITES if s != "__input__")
    cumulative = {
        "input": _nll_with(("__input__",)),
        "input+early": _nll_with(("__input__", *early_rest)),
        "input+early+deeper": full_nll,  # identical to the full stressed condition
    }
    decomposition = {
        "base_nll_clean_calib": base_nll,
        "full_stressed_calib_nll": full_nll,
        "total_damage": full_nll - base_nll,
        "single_site_delta_nll": single_site,
        "cumulative_nll": cumulative,
        "cumulative_damage": {k: v - base_nll for k, v in cumulative.items()},
    }
    writer.record_metric(
        "mechanism_decomposition[minmax][W4A4][clean_eval]",
        decomposition,
        Provenance.SIMULATED,
        note=SIM_NOTE + "; stressed qparams substituted per site/stage",
    )
    print(f"artifact: {writer.finalize()}")
    return {"fp32": fp32, "results": results, "decomposition": decomposition}


def main() -> int:
    if SUMMARY_PATH.exists():
        print(f"REFUSED: {SUMMARY_PATH} already exists — the D study is a single run.")
        return 2
    per_seed = {seed: run_seed(seed) for seed in SEEDS}

    def metric(seed, obs, cfg, cond, name):
        return per_seed[seed]["results"][f"{obs}|{cfg}|{cond}"][name]

    summary: dict = {
        "design": "ADR-012 (+ addenda 2, 4, 5); stress = Gate-v3 impulse 7-sigma",
        "provenance": "all quantized metrics SIMULATED (fake-quant policy v1); fp32 measured",
    }

    # Q1: robustness in the primary condition (stressed->clean, W4A4).
    q1 = {}
    for obs in ROBUST_OBSERVERS:
        improvements = [
            metric(s, "minmax", PRIMARY, "stressed->clean", "nll")
            - metric(s, obs, PRIMARY, "stressed->clean", "nll")
            for s in SEEDS
        ]
        acc_deltas = [
            metric(s, obs, PRIMARY, "stressed->clean", "accuracy")
            - metric(s, "minmax", PRIMARY, "stressed->clean", "accuracy")
            for s in SEEDS
        ]
        q1[obs] = {
            "mean_nll_improvement_vs_minmax": float(np.mean(improvements)),
            "per_seed_nll_improvement": improvements,
            "favorable_seeds": int(sum(i > 0 for i in improvements)),
            "mean_accuracy_delta": float(np.mean(acc_deltas)),
            "per_seed_accuracy_delta": acc_deltas,
            "confirmed": bool(
                np.mean(improvements) > Q1_NLL_IMPROVEMENT
                and sum(i > 0 for i in improvements) >= Q1_MIN_FAVORABLE_SEEDS
                and np.mean(acc_deltas) >= -Q1_MAX_ACCURACY_LOSS
            ),
        }
    summary["Q1_robustness_primary_condition"] = q1

    # Q2: clean-data non-inferiority, per configuration (never pooled).
    q2 = {}
    for obs in ROBUST_OBSERVERS:
        q2[obs] = {}
        for cfg in CONFIGS:
            worse_by = [
                metric(s, obs, cfg.label, "clean->clean", "nll")
                - metric(s, "minmax", cfg.label, "clean->clean", "nll")
                for s in SEEDS
            ]
            q2[obs][cfg.label] = {
                "mean_nll_worse_than_minmax": float(np.mean(worse_by)),
                "per_seed": worse_by,
                "non_inferior": bool(np.mean(worse_by) <= Q2_NLL_TOLERANCE),
            }
    summary["Q2_clean_non_inferiority"] = q2

    # Q3: power-of-two cost (measurement only, per config and checkpoint).
    q3 = {}
    for cfg in CONFIGS:
        q3[cfg.label] = {
            cond: {
                f"seed{s}_nll_cost_vs_minmax": metric(s, "pow2", cfg.label, cond, "nll")
                - metric(s, "minmax", cfg.label, cond, "nll")
                for s in SEEDS
            }
            for cond in ("clean->clean", "stressed->clean")
        }
    summary["Q3_pow2_cost_measurement_only"] = q3

    # Cross-seed observer-ranking stability (reported, not gated).
    order = list(OBSERVERS)
    rankings = {
        s: sorted(order, key=lambda o: metric(s, o, PRIMARY, "stressed->clean", "nll"))
        for s in SEEDS
    }
    rho = {}
    for a in SEEDS:
        for b in SEEDS:
            if a < b:
                ra = [rankings[a].index(o) for o in order]
                rb = [rankings[b].index(o) for o in order]
                rho[f"seed{a}-seed{b}"] = float(spearmanr(ra, rb).statistic)
    summary["ranking_stability_primary_condition"] = {
        "per_seed_ranking_best_to_worst": {f"seed{s}": rankings[s] for s in SEEDS},
        "spearman": rho,
    }

    # Mechanism decomposition attribution across seeds.
    summary["mechanism_decomposition"] = {
        f"seed{s}": {
            "total_damage": per_seed[s]["decomposition"]["total_damage"],
            "cumulative_damage": per_seed[s]["decomposition"]["cumulative_damage"],
            "single_site_delta_nll": per_seed[s]["decomposition"]["single_site_delta_nll"],
        }
        for s in SEEDS
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsummary: {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
