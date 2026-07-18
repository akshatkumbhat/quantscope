# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Every experimental result below is provenance-labeled in its artifacts
(measured / simulated / estimated); see `docs/REPORT.md` for the
findings and `docs/DECISIONS.md` for the full ADR audit trail.

## [0.1.0] — 2026-07-18

First complete release: every mandatory definition-of-done capability
implemented, verified in CI (Python 3.11/3.12, pinned torch 2.2.2),
and clean-clone checked.

### Added

- Affine quantization core (numpy-only): symmetric/asymmetric,
  per-tensor/per-channel, configurable bit widths, saturation,
  power-of-two scales; `torch_2_2` compatibility policy (ADR-011).
- Calibration observers: MinMax, Percentile(0.1/99.9), MSE-grid,
  PowerOfTwo (round-up).
- FX graph mode PTQ with measured INT8 CPU accuracy and sizes.
- Fake-quant simulation policy v1 (uniform / per-group /
  precomputed-parameter entry points).
- Texture-10 benchmark with frozen recipe and preregistered
  acceptance gates; impulse-stress mechanism.
- Sensitivity ablation + exhaustive 256-config mixed-precision sweeps
  with exact Pareto frontiers. Headline negative finding: one-shot
  ablation rankings did not guide joint assignment (ADR-010).
- Graph-anchored backend parity vs Torch FX INT8: qparams exact,
  0.0000 reference↔real-INT8 prediction disagreement on 3 seeds; two
  named compatibility findings (ADR-011).
- Observer-policy study under preregistered stress gates (two
  recorded gate failures, one pass); narrow finding:
  input/early-activation calibration robustness (ADR-012).
- Fixed-quantization-specification W4A4 QAT (Torch-native clipped
  STE, bit-exact forward parity): mean −0.0437 NLL vs PTQ, +1.60 pp,
  3/3 checkpoints (ADR-013).
- Analytical hardware cost model: schema-v1 fictional profile,
  deterministic accounting, component-wise estimated costs, budget
  recommendations, weight-bits-proxy comparison (ADR-014).
- Numerical-regression harness: baseline schema with canonical
  digests, four comparators, exit-code classification, deterministic
  diffs, CI smoke check (ADR-015).
- Deterministic report builder (figures + SHA-256 manifest) and the
  findings report `docs/REPORT.md`.
- Rigor amendments (ADR-016): FP32-finetune confound control (QAT
  claim upheld 3/3), FashionMNIST direction replication (2/2),
  paired bootstrap 95% CIs (all headline deltas exclude zero),
  ±50% cost-coefficient sensitivity (compute ratio is the
  load-bearing assumption), related-work positioning, mypy clean and
  enforced in CI.

### Fixed

- `src/quantscope/data` package was silently excluded by an
  unanchored `data/` gitignore rule (clones and CI were broken while
  local work passed); artifact-directory ignores are now root-anchored.
- CI reproducibility: ruff pinned to the repo's formatting minor and
  `known-first-party` declared; CI installs the validated torch
  2.2.2 environment instead of floating to latest.
- Cross-platform baseline tolerance: ReLU-site scales reclassified to
  the torch-derived family after a quantified few-ulp cross-BLAS
  difference (caught by the regression harness itself).
