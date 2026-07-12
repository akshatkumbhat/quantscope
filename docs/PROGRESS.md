# Project Progress

## Current phase

Phase 2 — Numerical quantization core

## Completed

- [x] Repository directories
- [x] Python package configuration
- [x] Initial Claude instructions
- [x] Project specification (`docs/PROJECT_SPEC.md`)
- [x] Environment repair (2026-07-11):
  - Root causes: shell resolved base-anaconda Python 3.9 instead of the
    `quantscope` env (3.11.13); `cli.py`/`__main__.py`/`test_cli.py`
    contained literal `\"` escape artifacts (invalid syntax).
  - Editable install with dev extras succeeds in the `quantscope` env.
  - `quantscope --help` / `version-info`, pytest, and ruff all pass.
- [x] PyTorch environment inspection (torch 2.2.2 / torchvision 0.17.2,
  Intel macOS, CPU-only; FX graph mode selected — ADR-006)
- [x] numpy<2 pin for torch 2.2.2 interop (ADR-005)
- [x] Architecture documented (`docs/architecture.md`)
- [x] CI workflow, pre-commit config, Makefile, LICENSE
- [x] Fictional hardware profile (`configs/hardware/generic_edge_npu.yaml`)

- [x] Affine quantization core (`quantization/affine.py`): integer ranges,
  symmetric/asymmetric scale+zero-point, per-tensor/per-channel,
  quantize/dequantize/fake-quantize, power-of-two scale approximation,
  typed `QuantParams` metadata. All arithmetic here is **simulated**.
- [x] 47 unit tests for the affine core.
- [x] Phase 3 — numerical-error metrics (`analysis/metrics.py`): MSE,
  SQNR (dB), cosine similarity, max-abs error, saturation rate, combined
  `ErrorMetrics` with JSON-serializable output; documented degenerate-case
  behavior (zero signal, zero noise, zero vectors).
- [x] Phase 3 — observers (`observers/`): streaming `CalibrationObserver`
  base + `MinMaxObserver` baseline, `PercentileClippingObserver`,
  `PowerOfTwoScaleObserver` (round-up snapping, zero-point recomputed),
  `MSEGridSearchObserver` (ADR-007). Empty/NaN calibration rejected.
- [x] 90 unit tests passing total; ruff lint + format clean; CLI works.

- [x] Phase 4 — configs, training, and PTQ:
  - Typed pydantic configs (`config/schemas.py`), YAML loading, strict
    validation, frozen models; `Provenance` label enum (ADR-004).
  - Utilities: seeding, environment capture (packages/git/engines),
    `RunWriter` artifact I/O that refuses unlabeled metrics.
  - FX-traceable `TinyCNN`; deterministic synthetic dataset (sinusoid
    patterns per class, no downloads); CPU FP32 train/eval loop.
  - FX-graph-mode PTQ (`quantization/ptq.py`, fbgemm engine): calibrate,
    convert, measured INT8 CPU accuracy + serialized sizes vs FP32.
  - CLI: `env-info`, `train-fp32`, `ptq`; quickstart config runs
    end-to-end in seconds (`configs/quickstart/quick.yaml`).
- [x] 116 tests passing (incl. PTQ integration test); ruff clean.

## Current phase (updated)

Benchmark plan step B — per-group W4A4 ablation, sensitivity ranking, and
(gate permitting) exhaustive 256-config mixed-precision search.

Step A closed **partial pass** (ADR-008 addendum 2): benchmark
discriminates bit widths (mean W4A4 −2.2 pp / NLL +64% across 3 seeds;
W8A8 ≈ FP32 with NLL effect +4e-5), with two recorded deviations — mean
FP32 96.22% vs the ≤96% condition, and the W8A8 mean-margin direction
failed on all seeds (margin rose slightly; see prospective metric note).
Deliverables: Texture-10 generator, BottleneckResNet (~20k params, 8
distinct ReLU sites), evaluate_detailed (accuracy/NLL/margin), simulation
policy v1 (quantization/simulate.py), texture-bench CLI, acceptance
checker script. FP32 checkpoints for seeds 0-2 live under
runs/validation/ (not committed).

## Blocked

None.

## B status (2026-07-12)

- [x] B1: eight-group partition, per-group simulated quantization,
  prediction-flips metric, `ablate` CLI; unit tests for partition
  coverage and targeted-group isolation.
- [x] B2: 3-seed ablation ran; stop gate evaluated —
  **STOP before C** (ADR-008 addendum 3). Sensitivity is meaningful
  (max mean ΔNLL +0.014) and non-uniform (max/median 3.76) but
  unstable across seeds (Spearman ≤ 0.405; top-2 reproduced 1/3).
- [ ] B3 (exhaustive 256-config search): deferred until a stable
  ranking exists.

## Next actions

1. Await approval of the scoped generator change: tighter inter-class
   separations in `texture10._class_components` to raise per-group
   effect sizes above inter-seed variance (target: FP32 88-94% and
   stability criterion passing), with its own iteration budget.
2. If approved and passing: rerun B2 gate, then B3.

## Known observations

- Quickstart accuracy saturates at 1.0 (task intentionally easy);
  differences will show at lower bit widths and in per-layer metrics.
- INT8 size compression on TinyCNN is ~1.5x, not the asymptotic ~4x:
  serialization overhead dominates at this model size (documented in the
  integration test).
