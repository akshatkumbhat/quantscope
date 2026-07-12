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

## Current phase (updated)

Phase 4 — FP32 training/evaluation and PTQ integration is next.

## Blocked

None.

## Next actions

1. Typed experiment config schemas (pydantic) and artifact I/O with
   measured/simulated/estimated provenance labels.
2. Tiny CPU-friendly model + synthetic/small dataset plumbing.
3. FP32 train/eval loop, then FX-graph-mode PTQ using these observers.
