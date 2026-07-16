# QuantScope

Neural-network quantization, sensitivity analysis, and numerical
debugging on CPU — with preregistered experiments, provenance-labeled
artifacts, and an honest findings report.

Findings (including the negative results and failed experiment gates)
are summarized in **[docs/REPORT.md](docs/REPORT.md)**; decisions and
experiment history live in [docs/DECISIONS.md](docs/DECISIONS.md).

## What is implemented

**Quantization core** (`quantscope.quantization.affine`) — standalone
affine quantization arithmetic: symmetric/asymmetric schemes,
per-tensor and per-channel granularity, configurable integer bit
widths, saturation/clipping, power-of-two scale approximation, typed
`QuantParams`. All arithmetic here is *simulated* (fake-quant in
FP32), never presented as integer execution.

**Calibration observers** (`quantscope.observers`) — streaming
`MinMaxObserver` baseline plus three custom policies:
`PercentileClippingObserver` (0.1/99.9), `MSEGridSearchObserver`,
and `PowerOfTwoScaleObserver` (round-up, zero-point recomputed).

**Post-training quantization** (`quantscope.quantization.ptq`) — FX
graph mode with the fbgemm engine: calibrate, convert, and report
*measured* INT8 CPU accuracy and serialized model sizes vs FP32.

**Simulation policy v1** (`quantscope.quantization.simulate`) —
uniform, per-group, and precomputed-parameter fake-quant simulation
of arbitrary bit widths (e.g. W4A4), with activation calibration
against the weight-quantized model.

**Numerical analysis** (`quantscope.analysis`) — per-layer error
metrics (MSE, SQNR dB, cosine similarity, max-abs error, saturation
rate) with documented degenerate-case behavior, plus the preregistered
stress-gate decision logic.

**Sensitivity analysis & mixed-precision search**
(`quantscope.sensitivity`, `quantscope.search`) — per-group W4A4
ablation, exhaustive 256-configuration sweeps, exact Pareto frontiers,
and predeclared search-strategy analysis. Headline (negative) finding:
one-at-a-time ablation rankings failed to guide joint assignment on
this benchmark — see the report before reusing that heuristic.

**Backend parity** (`quantscope.quantization.parity`) — a
graph-anchored comparison of QuantScope arithmetic against Torch FX
INT8 (Torch 2.2 compatibility mode), validated on three checkpoints
with two named compatibility findings (127.5-vs-127 symmetric-scale
denominator; division-precision tie resolution).

**Calibration-robustness study** — paired impulse-contamination
stress design with preregistered pass/fail gates (two failed gates
recorded as failed; the third passed) and a four-observer comparison.
Narrow conclusion: percentile/MSE-grid observers protect
input/early-activation calibration at 4-bit activations; not a
network-wide superiority claim.

**Quantization-aware training (simulated W4A4)**
(`quantscope.quantization.qat`) — fixed-quantization-specification QAT
using fake-quant policy v1 and a Torch-native clipped straight-through
estimator with bit-exact forward parity against the NumPy reference.
Activation qparams are frozen from PTQ calibration; per-channel weight
qparams are recomputed from the current weights under a frozen
quantization specification. Preregistered three-checkpoint result
(ADR-013, all *simulated*): mean NLL improvement vs PTQ **0.0437**,
mean accuracy recovery **1.60 pp**, improvement on all three
checkpoints; no checkpoint reached FP32 quality. Fine-tuning took
~151 s per checkpoint, measured CPU wall-clock on the project machine
— no real INT4 execution, deployment speedup, or accelerator
performance is implied. Details: ADR-013 in
[docs/DECISIONS.md](docs/DECISIONS.md); artifacts under
`runs/validation-012/texture-a-seed{0,1,2}-qat-w4a4/`.

**Reproducible artifacts & reporting** (`quantscope.utilities`,
`quantscope.reporting`) — every run writes config, environment, and
metrics with mandatory measured/simulated/estimated provenance labels;
`scripts/build_report.py` regenerates the report figures
byte-deterministically from artifacts with a hash manifest.

**Analytical hardware cost model** (`quantscope.hardware`, ADR-014) —
a validated schema-v1 profile format (Pydantic, fictional-by-
declaration, duplicate-pair detection), deterministic model accounting
with a declared no-cache single-read/single-write-per-tensor traffic
assumption and explicit exclusion lists, and transparent
component-wise *estimated* costs (compute / weight-memory /
activation-memory / overhead) normalized to all-INT8 = 1.0. Generates
checkpoint-specific mixed-precision recommendations at normalized
budgets 0.60/0.75/0.90 from the frozen B3 sweeps, and quantifies how
the profile changes conclusions vs the old weight-bits proxy
(Spearman ρ 0.886; Pareto membership Jaccard 0.30–0.61;
recommendations changed in 7 of 9 checkpoint×budget cells). The
`generic_edge_npu` profile is fictional — its coefficients are
assumptions, not calibrated hardware facts, and costs cover the
modeled quantizable workload only.

## Not implemented (honest gaps)

- **Numerical-regression test harness** — not built.
- Deferred experiment appendices: W3A3 stress test; Q4 sim↔backend
  observer comparison at W8A8 (rationale in ADR-012 addendum 6).

## Quickstart

Requires Python 3.11+ and a CPU; no downloads, no network access.

```bash
pip install -e ".[dev]"

# End-to-end quick mode (~1 minute): FP32 train + measured INT8 PTQ.
quantscope train-fp32 --config configs/quickstart/quick.yaml
quantscope ptq --config configs/quickstart/quick.yaml

# Texture-10 benchmark seed: FP32 (measured) vs W8A8/W4A4 (simulated).
quantscope texture-bench --seed 0 --freq-step 0.12 --output-dir runs

# Regenerate the report figures from existing artifacts.
python scripts/build_report.py
```

Artifacts land under `runs/<run-name>/` as `config.json`,
`environment.json`, and `metrics.json`; every metric carries a
provenance label and unlabeled metrics are refused.

### CLI

| Command | What it does |
| --- | --- |
| `quantscope env-info` | interpreter/package/quant-engine info |
| `quantscope train-fp32 -c <cfg>` | FP32 baseline training (measured) |
| `quantscope ptq -c <cfg>` | FX-graph INT8 PTQ (measured, CPU) |
| `quantscope texture-bench --seed N` | benchmark A: FP32 vs simulated W8A8/W4A4 |
| `quantscope ablate --seed N` | per-group W4A4 sensitivity ablation |
| `quantscope sweep --seed N` | exhaustive 256-config mixed-precision sweep |
| `quantscope backend-parity --seed N` | QuantScope↔Torch INT8 parity harness |
| `quantscope hw-validate --profile <yaml>` | validate a schema-v1 hardware profile |
| `quantscope hw-score --bits 4,8,…` | component-wise estimated cost of one assignment |

Training/benchmark commands are slow-path (minutes); core tests and
quick mode stay fast.

## Provenance rules

Every reported number is one of:

- **measured** — actually executed and observed (FP32 evaluation, real
  INT8 CPU accuracy/size in PTQ and parity),
- **simulated** — fake-quant simulation policy v1, not integer
  execution,
- **estimated** — analytical proxy (normalized weight-bits cost).

CPU latency is never presented as accelerator latency; no number in
this repository is NPU or hardware performance.

## Development

```bash
make test-fast   # core suite (CPU, offline, <1 min)
make test        # includes slow integration tests
make check       # ruff lint + format + pytest
```

CI (GitHub Actions) runs lint, format check, the fast suite, and a
CLI smoke test on Python 3.11 and 3.12.

## Disclaimer

QuantScope does not use, emulate, or claim compatibility with
proprietary Quadric technology.

The included hardware profile is fictional and illustrative. CPU
latency must not be presented as NPU latency. Simulated INT4
arithmetic and analytical hardware costs must be labeled clearly.
