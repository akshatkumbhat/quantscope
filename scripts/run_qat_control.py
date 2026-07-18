#!/usr/bin/env python
"""ADR-016 Part A: the QAT confound control.

Fine-tunes each validation checkpoint in plain FP32 with the IDENTICAL
frozen ADR-013 recipe (same seeds, batch order, optimizer, schedule,
epochs; no fake quantization anywhere), then applies the standard PTQ
procedure to the fine-tuned model. Compares against the recorded QAT
results under the predeclared interpretation:

  UPHELD    iff QAT W4A4 NLL < control-PTQ W4A4 NLL on >= 2/3
            checkpoints AND mean improvement >= 0.01
  DOWNGRADED otherwise (gains attributable to extended training).

All W4A4 numbers simulated (fake-quant policy v1); FP32 measured.
Original ADR-013 artifacts are never rewritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from quantscope.benchmark import benchmark_config, texture10_calibration
from quantscope.config import Provenance
from quantscope.data.synthetic import build_datasets
from quantscope.evaluation.loop import evaluate_detailed
from quantscope.models.tiny_cnn import build_model
from quantscope.quantization.qat import QATRecipe, fp32_finetune
from quantscope.quantization.simulate import SimQuantConfig, simulate_quantized
from quantscope.utilities import RunWriter

SEEDS = (0, 1, 2)
VALIDATION_DIR = Path("runs/validation-012")
SUMMARY_PATH = VALIDATION_DIR / "qat-control-summary.json"
QAT_SUMMARY = VALIDATION_DIR / "qat-study-summary.json"
SIM_NOTE = "fake-quant simulation policy v1; not integer execution"
FROZEN_LR = 3e-4  # the ADR-013 frozen recipe, unchanged
EPOCHS = 10


def main() -> int:
    if SUMMARY_PATH.exists():
        print(f"REFUSED: {SUMMARY_PATH} exists — the control arm runs once.")
        return 2
    qat = json.loads(QAT_SUMMARY.read_text())["per_seed"]

    per_seed = {}
    for seed in SEEDS:
        config = benchmark_config(seed=seed, output_dir=str(VALIDATION_DIR), freq_step=0.12)
        checkpoint = VALIDATION_DIR / f"texture-a-seed{seed}-fp32" / "model.pt"
        model = build_model(config.model)
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        train_set, eval_set = build_datasets(config.data, config.model)
        calib = texture10_calibration(config)

        recipe = QATRecipe(learning_rate=FROZEN_LR, epochs=EPOCHS, seed=100 + seed)
        control, history = fp32_finetune(model, train_set, recipe)
        control_fp32 = evaluate_detailed(control, eval_set)
        control_ptq = evaluate_detailed(
            simulate_quantized(control, calib, SimQuantConfig(4, 4)), eval_set
        )

        recorded = qat[f"seed{seed}"]
        row = {
            "control_fp32_nll": control_fp32["nll"],
            "control_fp32_accuracy": control_fp32["accuracy"],
            "control_ptq_nll": control_ptq["nll"],
            "control_ptq_accuracy": control_ptq["accuracy"],
            "recorded_qat_nll": recorded["qat_nll"],
            "recorded_qat_accuracy": recorded["qat_accuracy"],
            "recorded_ptq_nll": recorded["ptq_nll"],
            "recorded_fp32_nll": recorded["fp32_nll"],
            "qat_improvement_over_control": control_ptq["nll"] - recorded["qat_nll"],
            "control_improvement_over_original_ptq": recorded["ptq_nll"] - control_ptq["nll"],
        }
        per_seed[f"seed{seed}"] = row

        writer = RunWriter(config, kind="qat-control")
        torch.save(control.state_dict(), writer.run_dir / "model_control_fp32.pt")
        writer.record_metric(
            "recipe",
            {
                "learning_rate": FROZEN_LR,
                "epochs": EPOCHS,
                "seed": recipe.seed,
                "identical_to_qat_arm": "same loop/seeding/batch order; no fake quant",
            },
            Provenance.MEASURED,
        )
        writer.record_metric("epoch_train_loss", history.epoch_train_loss, Provenance.MEASURED)
        writer.record_metric("control_fp32_eval", control_fp32, Provenance.MEASURED)
        writer.record_metric(
            "control_ptq_w4a4_eval", control_ptq, Provenance.SIMULATED, note=SIM_NOTE
        )
        writer.record_metric("comparison", row, Provenance.SIMULATED, note=SIM_NOTE)
        print(
            f"seed{seed}: control FP32 nll {control_fp32['nll']:.4f} "
            f"acc {control_fp32['accuracy']:.4f} | control-PTQ nll {control_ptq['nll']:.4f} "
            f"acc {control_ptq['accuracy']:.4f} | QAT nll {recorded['qat_nll']:.4f} "
            f"-> QAT better by {row['qat_improvement_over_control']:+.4f}"
        )
        print(f"artifact: {writer.finalize()}")

    gaps = [per_seed[f"seed{s}"]["qat_improvement_over_control"] for s in SEEDS]
    favorable = sum(g > 0 for g in gaps)
    upheld = favorable >= 2 and float(np.mean(gaps)) >= 0.01
    summary = {
        "design": "ADR-016 Part A: FP32-finetune control for ADR-013 QAT",
        "provenance": "W4A4 simulated; FP32 measured; ADR-013 artifacts untouched",
        "per_seed": per_seed,
        "qat_vs_control_nll_gaps": gaps,
        "favorable_checkpoints": favorable,
        "mean_gap": float(np.mean(gaps)),
        "verdict": "UPHELD" if upheld else "DOWNGRADED",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsummary: {SUMMARY_PATH}")
    print(
        f"VERDICT: {summary['verdict']} (favorable {favorable}/3, "
        f"mean gap {summary['mean_gap']:+.4f}, threshold 0.01)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
