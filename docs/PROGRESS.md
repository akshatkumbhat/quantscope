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
- [x] B3 (reframed deliverable, ADR-010): exhaustive 256-config sweeps
  on all three freq_step=0.12 checkpoints; exact Pareto frontiers;
  predeclared search analysis. **Methodological success; substantive
  negative finding**: one-shot ablation rankings did not guide search
  even within their own checkpoint (random 32-eval search beat the
  sensitivity path on all seeds; greedy joint-effect search worked);
  Pareto sets barely overlap across checkpoints (Jaccard 0.065–0.161).

## Generator phase status (2026-07-12)

FAILED at candidate screening (ADR-009 addendum). freq_step candidates
0.20/0.15/0.12 on dev seed 7 all left FP32 at ~96% (target ~90-93%);
the 18° orientation step dominates class identity. No candidate frozen
against that target. A later diagnostic + pre-registered re-scoped
validation at 0.12 also failed the decisive stability gate (ADR-009
addenda 2–4); B2's checkpoint-conditioned-sensitivity conclusion stands.

## C status (2026-07-13)

- [x] C: graph-anchored backend parity — **PASS on all 3 checkpoints**
  (ADR-011 + addendum). qparams exact under torch_2_2 policy; strict
  sim↔reference residuals fully localized to float32-vs-float64
  rounding-tie resolution (integer final-code steps; ≤0.5%
  prediction disagreement; accuracy/NLL aligned); reference↔real-INT8
  prediction disagreement 0.0000 on every seed. Two named
  compatibility findings recorded (127.5-vs-127 symmetric scale;
  division-precision tie resolution, systematic at requant nodes).

## D status (2026-07-14)

- [x] ADR-012 approved: observer-policy comparison study design
  (stress-design gate before validation seeds).
- [x] Stress-design Gate v1 (dev seed 7): **FAILED** on site coverage
  at both 6σ and 10σ (ADR-012 addendum). Mechanism finding: pixel
  impulses contaminate input/early ranges but attenuate before deeper
  sites; behavioral damage criterion passed at both magnitudes.
- [x] Impulse Stress Gate v2 (fresh dev seed 8, 6σ, preregistered
  ADR-012 addendum 2): **FAILED** on input expansion, 1.96× vs 2.0×
  (addendum 3). Behavioral criterion passed 5.6× over gate. Diagnosis:
  arithmetic tension between 6σ impulses and ~3.06σ clean-input
  extremes. Recorded failed as written; no post-hoc adjustment.
- [x] User decision: continue the impulse family with **Gate v3**,
  preregistered as ADR-012 addendum 4 — sole change 6σ → 7σ, 2.0×
  input threshold retained, fresh dev seed 6 (lowest unused seed whose
  streams touch no validation material), stress seed 1006, one attempt
  only, no fallback. Gate logic extracted to
  `quantscope.analysis.stress_gate` with unit tests; runner
  `scripts/check_stress_gate_v3.py` writes a labeled artifact and
  refuses a second attempt.
- [x] Seed-6 FP32 dev checkpoint trained (`runs/gen-dev6/`, frozen
  0.12 recipe; FP32 95.1% measured).
- [x] Impulse Stress Gate v3 executed once: **PASSED** (ADR-012
  addendum 5). Input expansion 2.43× (≥2.0×), early-site reach 4/4,
  behavioral +0.1925 NLL (9.6× gate), pairing intact. Artifact:
  `runs/gen-dev6/texture-a-seed6-stress-gate-v3/`.
- [x] D observer-policy study ran once on validation seeds 0/1/2
  (ADR-012 addendum 6). **Q1 CONFIRMED for percentile and MSE-grid**
  in the primary stressed→clean W4A4 condition (mean ΔNLL vs MinMax
  +0.070 / +0.073, favorable 3/3 seeds, accuracy +3.2 pp) — but the
  conclusion is narrow: the mechanism decomposition localizes ≥95% of
  MinMax damage to the input observer, so this is
  **input/early-activation calibration robustness**, not
  network-wide observer superiority. Q2 non-inferior in all six
  cells (robust observers slightly better than MinMax on clean data
  at A4). Q3: pow2 round-up is catastrophic under stressed
  calibration at A4 (+0.67 to +2.9 NLL), negligible at W8A8.
  Ranking stability: robust pair swaps 1st/2nd across seeds;
  minmax/pow2 tail stable. Evidence caveats recorded: ReLU-site
  saturation is dominated by exact zeros at qmin (clipping
  interpretation restricted to the input site); every quantized
  number is fake-quant simulation policy v1, not integer execution.
  Artifacts: `runs/validation-012/texture-a-seed{0,1,2}-observer-study/`,
  `runs/validation-012/observer-study-summary.json` (not committed).
- [x] Q4 (sim_custom ↔ backend-matched at W8A8) **deferred** to the
  optional appendix list: non-gating by design; C already validated
  backend-matched W8A8; D shows W8A8 observer differences ≤ ~0.001
  NLL; little marginal value now.

**Plan step D is complete.**

## Reporting phase status (2026-07-14)

- [x] Report figures generated deterministically from artifacts via
  `quantscope.reporting` + `scripts/build_report.py`: per-checkpoint
  B3 Pareto frontiers (never averaged; Jaccard overlap on-figure),
  D W4A4 factorial (per-checkpoint values), mechanism decomposition,
  Q3 pow2 cost. Field-level provenance labels on every figure and in
  `docs/report/figures/manifest.json` (source paths + SHA-256).
  Figures are small (~67–130 KB) and committed under
  `docs/report/figures/`; byte-identical on regeneration.
- [x] `docs/REPORT.md`: honest findings summary — B3 negative finding
  (scoped: one-at-a-time ablation rankings failed to guide joint
  assignment on this benchmark, not a general claim), C parity pass
  scoped to Torch-2.2-compatible arithmetic on this graph/config with
  the two named compatibility findings, D narrow positive
  (input/early-activation calibration robustness; pow2 result scoped
  to the frozen round-up policy under contaminated calibration), full
  gate v1→v2→v3 history with failures at equal prominence, both
  evidence caveats, reproduction instructions.
- [x] Tests: artifact loading fail-loud cases, manifest provenance
  labels, byte-deterministic outputs (synthetic fixtures, no runs/
  dependency).
- Out of scope per approved package (untouched): README, CI, W3A3,
  Q4, pushing.

## Close-out status (2026-07-15)

- [x] README rewritten to match actual behavior: implemented features,
  quickstart/CLI verified, provenance rules, and an explicit
  honest-gaps section (QAT not built; hardware cost model is a stub —
  sweep cost is the estimated normalized weight-bits proxy and the
  fictional YAML profile is unconsumed; no numerical-regression
  harness).
- [x] CI green on Python 3.11 and 3.12 (run 29382201255). Five root
  causes fixed in sequence: floating ruff version (pinned 0.15.x);
  environment-dependent first-party import classification (explicit
  known-first-party); **src/quantscope/data was never committed** — an
  unanchored `data/` gitignore rule silently excluded the dataset
  generators, so every clone/CI was broken while local work passed
  (artifact-dir ignores now anchored to the repo root); first-ever
  lint/format of that package; and unpinned torch resolving 2.13,
  tripping the ADR-011 version guard in the graph-anchored parity
  simulator (CI now installs the validated torch 2.2.2 / torchvision
  0.17.2 CPU environment; the guard stays in place). Deprecated
  checkout/setup-python actions bumped.

## Next actions

1. Optional appendix list (deferred, run only if separately
   approved): W3A3 stress test; Q4 sim_custom ↔ backend-matched
   W8A8 comparison (rationale in ADR-012 addendum 6); re-validating
   the parity simulator on newer torch (would lift the ADR-011 guard
   deliberately, not incidentally).

## Known observations

- Quickstart accuracy saturates at 1.0 (task intentionally easy);
  differences will show at lower bit widths and in per-layer metrics.
- INT8 size compression on TinyCNN is ~1.5x, not the asymptotic ~4x:
  serialization overhead dominates at this model size (documented in the
  integration test).
