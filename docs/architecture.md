# QuantScope Architecture

Status: current as of v0.1.0 (2026-07-18). Keep in sync with
`docs/DECISIONS.md`; module-level docstrings are the source of truth
for details.

## Validated environment

- CPU-only; developed on macOS Intel, CI on Ubuntu (Python 3.11/3.12).
- torch 2.2.2 / torchvision 0.17.2 — the validated environment
  (ADR-011 pins the graph-anchored parity simulator to torch 2.2.x
  with a runtime guard; CI installs exactly this environment).
- numpy `>=1.26,<2` (ADR-005: torch 2.2.2 interop).
- Quantized engines: fbgemm (used), qnnpack/onednn/x86 available.
- FX graph mode is the PyTorch integration API (ADR-006).

## Layering

```
┌──────────────────────────────────────────────────────────────┐
│ cli.py (Typer): train-fp32 · ptq · ablate · sweep ·          │
│   backend-parity · texture-bench · hw-* · regression *       │
├──────────────────────────────────────────────────────────────┤
│ workflows: evaluation/ · quantization/(ptq, qat, parity) ·   │
│   sensitivity/ · search/ · benchmark.py                      │
├──────────────────────────────────────────────────────────────┤
│ cross-cutting verification & reporting:                      │
│   regression/ (baseline checks) · reporting/ (deterministic  │
│   figures + manifest) · analysis/ (metrics, stress gates)    │
├───────────────────────────────┬──────────────────────────────┤
│ torch adapters:               │ hardware/: profile schema v1 │
│   quantization/simulate.py    │   (pydantic), deterministic  │
│   (policy v1 fake-quant),     │   model accounting,          │
│   quantization/qat.py         │   component-wise estimated   │
│   (clipped-STE training path) │   cost model                 │
├───────────────────────────────┴──────────────────────────────┤
│ numerical core (numpy-only, framework-independent):          │
│   quantization/affine.py · analysis/metrics.py               │
├──────────────────────────────────────────────────────────────┤
│ foundation: config/ (pydantic schemas, Provenance labels) ·  │
│   utilities/ (RunWriter artifacts, seeding, env capture) ·   │
│   models/ · data/ · observers/                               │
└──────────────────────────────────────────────────────────────┘
```

## Package map

| Package | Responsibility | Key ADRs |
| --- | --- | --- |
| `quantization/affine.py` | Backend-independent affine math: schemes, granularity, saturation, pow2 scales | 002, 011 |
| `quantization/simulate.py` | Fake-quant simulation policy v1 (uniform / per-group / precomputed-params); weight-quantized calibration | 008, 012 |
| `quantization/qat.py` | Torch-native clipped-STE training path, bit-exact forward parity with the NumPy core; `qat_finetune` + `fp32_finetune` control | 013, 016 |
| `quantization/ptq.py` | FX graph mode PTQ: measured INT8 CPU accuracy + sizes | 006 |
| `quantization/parity.py` | Graph-anchored backend parity vs Torch FX INT8 (`torch_2_2` policy; version guard) | 011 |
| `observers/` | MinMax baseline; Percentile, MSE-grid, PowerOfTwo custom observers | 007, 012 |
| `analysis/` | Error metrics (MSE, SQNR, cosine, saturation) + preregistered stress-gate decision logic | 004, 012 |
| `sensitivity/` | Per-group W4A4 ablation and prediction utilities | 008, 010 |
| `search/` | Exhaustive 256-config sweeps, exact Pareto frontiers, search strategies | 010 |
| `hardware/` | Profile schema v1 (list-based coefficients, dual hashes), `GROUP_ORDER_V1` accounting, float64 component costs, budget recommendations | 014, 016 |
| `regression/` | Numerical-regression harness: baseline schema + digests, JSON Pointer rules, 4 comparators, exit codes 0/1/2, deterministic diffs, smoke artifact | 015 |
| `reporting/` | Deterministic report figures + hash manifest from artifacts | — |
| `evaluation/` | FP32 train/eval loops, detailed metrics | — |
| `benchmark.py` | Frozen Texture-10 benchmark recipe (seed streams: train s, eval s+1, calib s+2, probe s+3) | 008, 009 |
| `models/`, `data/` | TinyCNN, BottleneckResNet; synthetic + Texture-10 generators, impulse stress | 008, 012 |
| `config/`, `utilities/` | Pydantic schemas, `Provenance` enum, `RunWriter` (refuses unlabeled metrics), seeding, env capture | 004 |

## Invariants (enforced, not aspirational)

1. **Provenance**: every persisted metric carries measured / simulated
   / estimated; `RunWriter.record_metric` rejects unlabeled values;
   the regression harness fails on provenance changes even when the
   value is unchanged.
2. **Numerical core independence**: `affine.py` and `metrics.py` are
   numpy-only; torch paths (simulate, qat) are adapters with
   unit-tested bit-exact forward parity against the core.
3. **Determinism**: report figures, the smoke artifact, and diff
   artifacts are byte-deterministic; accounting and profiles carry
   SHA-256 digests recorded in downstream artifacts.
4. **Version honesty**: the parity simulator refuses torch != 2.2.x
   at runtime rather than silently producing unvalidated numbers.
5. **Experiment lifecycle**: preregister (ADR) → commit → run once →
   record results (including failures) → never rewrite artifacts.
   One-run guards in every study script enforce the "run once".
6. **Artifacts**: `runs/<name>/` gets `config.json`,
   `environment.json`, `metrics.json` (+ checkpoints); `runs/` is
   never committed; committed baselines live in `tests/baselines/`.

## Documentation map

- `README.md` — capabilities, quickstart, CLI, provenance rules.
- `docs/REPORT.md` — the findings report (related work, B/C/D, QAT +
  control, replication, CIs, cost model + sensitivity).
- `docs/DECISIONS.md` — ADR-001..016 with amendments and results;
  the experiment audit trail, failures included.
- `docs/PROGRESS.md` — chronological status log.
- `docs/PROJECT_SPEC.md` — original specification.
- `CHANGELOG.md` — release history.
