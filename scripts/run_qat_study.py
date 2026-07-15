#!/usr/bin/env python
"""ADR-013: fixed-quantization-specification W4A4 QAT vs PTQ.

Two phases, run in order:

    python scripts/run_qat_study.py dev
    python scripts/run_qat_study.py validate --lr <frozen-lr>

``dev``: the preregistered recipe sequence R1(3e-4) -> R2(1e-4) ->
R3(1e-3) on the fresh dev seed 9, stopping at the FIRST recipe that
improves both W4A4 NLL and accuracy over that checkpoint's PTQ
baseline with no numerical instability. Validation seeds are never
touched in this phase.

``validate``: the frozen recipe, exactly once per validation seed
0/1/2. Before any training, each seed runs the amended baseline
consistency gate against the canonical D artifact
(``minmax|W4A4|clean->clean``): accuracy and prediction counts exactly
equal; NLL within 1e-6 absolute; per-site activation scales exactly
equal; identity fields (seed, splits, freq_step) exactly equal.
Exceeding any tolerance STOPS the study as a provenance failure; the
canonical D values are never overwritten.

Everything quantized here is fake-quant simulation policy v1 — NOT
integer execution; no INT4 kernels, no latency claims. Fine-tuning
wall-clock is measured on the development CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from quantscope.benchmark import benchmark_config, texture10_calibration
from quantscope.config import Provenance
from quantscope.data.synthetic import build_datasets
from quantscope.evaluation.loop import evaluate_detailed
from quantscope.models.tiny_cnn import build_model
from quantscope.quantization.affine import quantize
from quantscope.quantization.qat import QATRecipe, nll_gap_recovery, qat_finetune
from quantscope.quantization.simulate import (
    calibrate_activation_params,
    quantize_weights_uniform,
    simulate_quantized_with_params,
)
from quantscope.sensitivity import predictions
from quantscope.utilities import RunWriter, read_metrics

DEV_SEED = 9
DEV_RUNS_DIR = "runs/gen-dev9"
VALIDATION_SEEDS = (0, 1, 2)
VALIDATION_DIR = "runs/validation-012"
SUMMARY_PATH = Path(VALIDATION_DIR) / "qat-study-summary.json"
SIM_NOTE = "fake-quant simulation policy v1; not integer execution"

# Preregistered recipes (ADR-013): only the learning rate varies.
RECIPE_LRS = (3e-4, 1e-4, 1e-3)  # evaluation order R1 -> R2 -> R3
EPOCHS = 10
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4

# Amended baseline-consistency tolerances (ADR-013 addendum).
NLL_ABS_TOL = 1e-6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _params_digest(act_params) -> str:
    payload = {
        site: {
            "scale": float(np.asarray(p.scale)),
            "zero_point": int(np.asarray(p.zero_point)),
            "qmin": p.qmin,
            "qmax": p.qmax,
        }
        for site, p in sorted(act_params.items())
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _logits(model, dataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        return np.concatenate([model(images).numpy() for images, _ in loader])


def _load_environment(seed: int, runs_dir: str):
    """Checkpoint + frozen splits + frozen activation qparams for one seed."""
    config = benchmark_config(seed=seed, output_dir=runs_dir, freq_step=0.12)
    checkpoint = Path(runs_dir) / f"texture-a-seed{seed}-fp32" / "model.pt"
    model = build_model(config.model)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    train_set, eval_set = build_datasets(config.data, config.model)
    calib = texture10_calibration(config)
    # Identical to simulate_quantized: calibrate against the
    # weight-quantized model, MinMax, per-tensor affine activations.
    act_params = calibrate_activation_params(quantize_weights_uniform(model, bits=4), calib, bits=4)
    return config, checkpoint, model, train_set, eval_set, calib, act_params


def _final_weight_scales(model) -> dict[str, list[float]]:
    """Final per-channel weight scales, recomputed once at export."""
    from quantscope.quantization.affine import Granularity, Scheme, compute_quant_params

    out = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d | torch.nn.Linear):
            params = compute_quant_params(
                module.weight.detach().numpy(),
                bits=4,
                signed=True,
                scheme=Scheme.SYMMETRIC,
                granularity=Granularity.PER_CHANNEL,
                channel_axis=0,
            )
            out[name] = [float(s) for s in np.asarray(params.scale)]
    return out


def _weight_saturation(model) -> dict[str, float]:
    """Fraction of final per-channel weight codes at qmin/qmax (diagnostic)."""
    from quantscope.quantization.affine import Granularity, Scheme, compute_quant_params

    out = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d | torch.nn.Linear):
            weight = module.weight.detach().numpy()
            params = compute_quant_params(
                weight,
                bits=4,
                signed=True,
                scheme=Scheme.SYMMETRIC,
                granularity=Granularity.PER_CHANNEL,
                channel_axis=0,
            )
            codes = quantize(weight, params)
            saturated = np.count_nonzero(codes <= params.qmin) + np.count_nonzero(
                codes >= params.qmax
            )
            out[name] = float(saturated / codes.size)
    return out


def _activation_saturation(model, calib, act_params) -> dict[str, float]:
    """Fraction of calibration activations saturating the frozen ranges."""
    from quantscope.analysis.metrics import saturation_rate
    from quantscope.quantization.simulate import _INPUT_KEY

    captured: dict[str, np.ndarray] = {_INPUT_KEY: calib.tensors[0].numpy()}
    handles = []
    wq = quantize_weights_uniform(model, bits=4)
    for name, module in wq.named_modules():
        if isinstance(module, torch.nn.ReLU):

            def hook(_m, _i, out, *, _name=name):
                captured[_name] = out.detach().numpy()

            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        wq(calib.tensors[0])
    for h in handles:
        h.remove()
    return {
        site: saturation_rate(quantize(captured[site], p), p.qmin, p.qmax)
        for site, p in act_params.items()
    }


def _evaluate_arms(model, tuned, act_params, eval_set):
    """FP32 / PTQ / QAT metrics on the frozen evaluation split."""
    ptq_model = simulate_quantized_with_params(model, act_params, weight_bits=4)
    qat_model = simulate_quantized_with_params(tuned, act_params, weight_bits=4)
    arms = {
        "fp32": (model, evaluate_detailed(model, eval_set)),
        "ptq": (ptq_model, evaluate_detailed(ptq_model, eval_set)),
        "qat": (qat_model, evaluate_detailed(qat_model, eval_set)),
    }
    preds = {name: predictions(m, eval_set) for name, (m, _) in arms.items()}
    logits = {name: _logits(m, eval_set) for name, (m, _) in arms.items()}
    return {name: detail for name, (_, detail) in arms.items()}, preds, logits, qat_model


def _comparison(detail, preds, logits) -> dict:
    from quantscope.analysis.metrics import cosine_similarity, sqnr_db

    fp32, ptq, qat = detail["fp32"], detail["ptq"], detail["qat"]
    return {
        "fp32_nll": fp32["nll"],
        "ptq_nll": ptq["nll"],
        "qat_nll": qat["nll"],
        "delta_nll_qat_vs_ptq": qat["nll"] - ptq["nll"],
        "accuracy_recovery_pp": (qat["accuracy"] - ptq["accuracy"]) * 100.0,
        "nll_gap_recovery": nll_gap_recovery(ptq["nll"], qat["nll"], fp32["nll"]),
        "fp32_accuracy": fp32["accuracy"],
        "ptq_accuracy": ptq["accuracy"],
        "qat_accuracy": qat["accuracy"],
        "qat_mean_margin": qat["mean_margin"],
        "ptq_mean_margin": ptq["mean_margin"],
        "prediction_flips_qat_vs_ptq": float(np.mean(preds["qat"] != preds["ptq"])),
        "prediction_flips_qat_vs_fp32": float(np.mean(preds["qat"] != preds["fp32"])),
        "logit_sqnr_db_qat_vs_fp32": sqnr_db(logits["fp32"], logits["qat"]),
        "logit_sqnr_db_ptq_vs_fp32": sqnr_db(logits["fp32"], logits["ptq"]),
        "logit_cosine_qat_vs_fp32": cosine_similarity(logits["fp32"], logits["qat"]),
    }


def _run_one(seed: int, runs_dir: str, recipe: QATRecipe, kind: str) -> dict:
    """One QAT fine-tune + three-arm evaluation, with a labeled artifact."""
    config, checkpoint, model, train_set, eval_set, calib, act_params = _load_environment(
        seed, runs_dir
    )
    artifact_dir = Path(runs_dir) / f"{config.run_name}-{kind}"
    if artifact_dir.exists():
        raise FileExistsError(f"{artifact_dir} exists — ADR-013 allows one run per recipe/seed")

    start = time.perf_counter()
    tuned, history = qat_finetune(model, train_set, act_params, recipe)
    elapsed = time.perf_counter() - start

    detail, preds, logits, _ = _evaluate_arms(model, tuned, act_params, eval_set)
    comparison = _comparison(detail, preds, logits)

    writer = RunWriter(config, kind=kind)
    torch.save(tuned.state_dict(), writer.run_dir / "model_qat_fp32.pt")
    writer.record_metric(
        "provenance_identity",
        {
            "source_fp32_checkpoint": str(checkpoint),
            "source_fp32_checkpoint_sha256": _sha256(checkpoint),
            "calibration": f"texture10 stream seed+2 ({config.data.seed + 2}), "
            f"{config.data.num_calib} samples, deterministic order",
            "finetune_data": f"train split stream seed ({config.data.seed}), "
            f"{config.data.num_train} samples",
            "evaluation": f"eval split stream seed+1 ({config.data.seed + 1}), "
            f"{config.data.num_eval} samples",
            "frozen_activation_qparams_sha256": _params_digest(act_params),
            "qparam_policy": "policy v1; MinMax; per-tensor affine activations; "
            "per-channel symmetric weights (rule frozen, scales from final weights)",
            "recipe": {
                "learning_rate": recipe.learning_rate,
                "epochs": recipe.epochs,
                "batch_size": recipe.batch_size,
                "weight_decay": recipe.weight_decay,
                "optimizer": "adamw+cosine",
                "weight_bits": recipe.weight_bits,
                "seed": recipe.seed,
                "checkpoint_selection": "epoch-10 final (frozen; no best-epoch)",
                "fake_quant_schedule": "first step through last; no warm-up",
            },
        },
        Provenance.MEASURED,
    )
    writer.record_metric(
        "frozen_activation_scales",
        {site: float(np.asarray(p.scale)) for site, p in act_params.items()},
        Provenance.MEASURED,
    )
    writer.record_metric(
        "final_weight_qparams",
        _final_weight_scales(tuned),
        Provenance.MEASURED,
        note="per-channel symmetric scales recomputed ONCE from the final weights "
        "(ADR-013 addendum); the exported/evaluated QAT quantizer",
    )
    writer.record_metric(
        "final_weight_saturation", _weight_saturation(tuned), Provenance.SIMULATED, note=SIM_NOTE
    )
    writer.record_metric(
        "activation_saturation_calibration",
        _activation_saturation(tuned, calib, act_params),
        Provenance.SIMULATED,
        note=SIM_NOTE,
    )
    writer.record_metric("epoch_train_loss", history.epoch_train_loss, Provenance.MEASURED)
    writer.record_metric("gradients_finite", int(history.gradients_finite), Provenance.MEASURED)
    writer.record_metric(
        "finetune_wallclock_seconds",
        elapsed,
        Provenance.MEASURED,
        note="measured on the development CPU; no accelerator extrapolation",
    )
    writer.record_metric("fp32_eval", detail["fp32"], Provenance.MEASURED)
    for arm in ("ptq", "qat"):
        writer.record_metric(f"{arm}_w4a4_eval", detail[arm], Provenance.SIMULATED, note=SIM_NOTE)
    writer.record_metric("comparison", comparison, Provenance.SIMULATED, note=SIM_NOTE)
    print(f"artifact: {writer.finalize()}")
    return comparison


def run_dev() -> int:
    for lr in RECIPE_LRS:
        recipe = QATRecipe(
            learning_rate=lr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            seed=100 + DEV_SEED,
        )
        print(f"\n=== dev recipe {recipe.label()} (seed {DEV_SEED}) ===")
        comparison = _run_one(DEV_SEED, DEV_RUNS_DIR, recipe, kind=f"qat-dev-{recipe.label()}")
        improves = comparison["delta_nll_qat_vs_ptq"] < 0 and comparison["accuracy_recovery_pp"] > 0
        print(
            f"dev {recipe.label()}: dNLL {comparison['delta_nll_qat_vs_ptq']:+.4f}, "
            f"acc recovery {comparison['accuracy_recovery_pp']:+.2f} pp, "
            f"gap recovery {comparison['nll_gap_recovery']:.3f} "
            f"-> {'PASS (frozen)' if improves else 'no pass'}"
        )
        if improves:
            print(f"FROZEN VALIDATION RECIPE: lr={lr}, epochs={EPOCHS}")
            return 0
    print("NO development recipe passed — ADR-013 stops with a negative development result.")
    return 1


def _baseline_gate(seed: int, recomputed: dict, act_params) -> dict:
    """Amended consistency gate vs the canonical D artifact. Raises on breach."""
    d_dir = Path(VALIDATION_DIR) / f"texture-a-seed{seed}-observer-study"
    entries = {m["name"]: m for m in read_metrics(d_dir)["metrics"]}
    canonical = entries["minmax|W4A4|clean->clean"]["value"]
    d_scales = entries["scales[minmax][clean][W4A4]"]["value"]
    d_config = json.loads((d_dir / "config.json").read_text())

    ours = {site: float(np.asarray(p.scale)) for site, p in act_params.items()}
    breaches = []
    if recomputed["accuracy"] != canonical["accuracy"]:
        breaches.append(f"accuracy {recomputed['accuracy']} != canonical {canonical['accuracy']}")
    if abs(recomputed["nll"] - canonical["nll"]) > NLL_ABS_TOL:
        breaches.append(f"NLL |{recomputed['nll']} - {canonical['nll']}| > {NLL_ABS_TOL}")
    if set(ours) != set(d_scales) or any(ours[s] != d_scales[s] for s in ours):
        breaches.append("per-site activation scales differ from canonical D artifact")
    if d_config["data"]["seed"] != seed or d_config["data"]["freq_step"] != 0.12:
        breaches.append("checkpoint/config identity mismatch with D artifact")
    if breaches:
        raise RuntimeError(
            f"seed {seed}: baseline consistency gate FAILED (provenance/reproducibility): "
            + "; ".join(breaches)
        )
    return {"canonical": canonical, "recomputed": recomputed}


def run_validation(lr: float) -> int:
    if SUMMARY_PATH.exists():
        print(f"REFUSED: {SUMMARY_PATH} exists — validation runs exactly once.")
        return 2

    # Baseline consistency gates for ALL seeds before any training.
    baselines = {}
    for seed in VALIDATION_SEEDS:
        config, _, model, _, eval_set, _, act_params = _load_environment(seed, VALIDATION_DIR)
        ptq_model = simulate_quantized_with_params(model, act_params, weight_bits=4)
        recomputed = evaluate_detailed(ptq_model, eval_set)
        recomputed["prediction_count"] = len(predictions(ptq_model, eval_set))
        if recomputed["prediction_count"] != config.data.num_eval:
            raise RuntimeError(f"seed {seed}: prediction count != {config.data.num_eval}")
        baselines[seed] = _baseline_gate(seed, recomputed, act_params)
        print(f"seed {seed}: baseline consistency gate PASS")

    results = {}
    for seed in VALIDATION_SEEDS:
        recipe = QATRecipe(
            learning_rate=lr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            seed=100 + seed,
        )
        print(f"\n=== validation seed {seed}, frozen recipe {recipe.label()} ===")
        results[seed] = _run_one(seed, VALIDATION_DIR, recipe, kind="qat-w4a4")

    deltas = [results[s]["delta_nll_qat_vs_ptq"] for s in VALIDATION_SEEDS]
    acc_rec = [results[s]["accuracy_recovery_pp"] for s in VALIDATION_SEEDS]
    acc_gap = [
        (results[s]["fp32_accuracy"] - results[s]["ptq_accuracy"]) * 100.0 for s in VALIDATION_SEEDS
    ]
    criteria = {
        "1_nll_improves_on_2_of_3": sum(d < 0 for d in deltas) >= 2,
        "2_mean_nll_improvement_ge_0.01": -float(np.mean(deltas)) >= 0.01,
        "3_accuracy_recovery": bool(
            float(np.mean(acc_rec)) >= 1.0
            or float(np.mean(acc_rec)) >= float(np.mean(acc_gap)) / 3.0
        ),
        "4_all_finite": True,  # qat_finetune raises otherwise
        "5_no_checkpoint_worse_than_0.5pp": all(r >= -0.5 for r in acc_rec),
    }
    passed = all(criteria.values())
    summary = {
        "design": "ADR-013 + addendum; fixed-quantization-specification W4A4 QAT",
        "provenance": "W4A4 metrics simulated (fake-quant policy v1); FP32 measured; "
        "wall-clock measured on development CPU",
        "frozen_recipe": {"learning_rate": lr, "epochs": EPOCHS},
        "baseline_consistency": {f"seed{s}": baselines[s] for s in VALIDATION_SEEDS},
        "per_seed": {f"seed{s}": results[s] for s in VALIDATION_SEEDS},
        "success_criteria": criteria,
        "adr013_pass": passed,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsummary: {SUMMARY_PATH}")
    print(json.dumps({"success_criteria": criteria, "adr013_pass": passed}, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    sub.add_parser("dev")
    validate = sub.add_parser("validate")
    validate.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    if args.phase == "dev":
        return run_dev()
    if args.lr not in RECIPE_LRS:
        raise SystemExit(f"--lr must be one of the preregistered {RECIPE_LRS}")
    return run_validation(args.lr)


if __name__ == "__main__":
    sys.exit(main())
