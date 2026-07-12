# QuantScope — Project Specification

## Objective

QuantScope is a hardware-aware neural-network quantization, sensitivity-analysis,
and numerical-debugging toolkit for edge accelerators, built as a portfolio
project demonstrating model-optimization engineering skills.

It quantizes small vision models, measures exactly what quantization does to
every layer, ranks layers by sensitivity, and recommends mixed-precision
configurations under analytical hardware constraints — with every reported
number honestly labeled as measured, simulated, or estimated.

## Scope

In scope:

- Small CNNs (and optionally a small transformer block) trained on small
  datasets (synthetic tensors, MNIST/CIFAR-10-class datasets) on CPU.
- INT4–INT8 simulated quantization; INT8 backend execution where PyTorch
  provides CPU kernels.
- Analytical (not measured) hardware cost modeling for a fictional edge NPU.

Out of scope:

- Real NPU deployment or vendor toolchains.
- Large models or datasets requiring GPUs.
- Claiming any result represents proprietary hardware (see Disclaimer).

## Functional requirements

### 1. FP32 baseline

- Train and evaluate a small FP32 model on CPU (quick mode: minutes).
- Persist checkpoints, resolved config, seed, and metrics (measured).

### 2. Standalone affine quantization core (backend-independent)

- Affine mapping: `real ≈ scale × (q − zero_point)`.
- Symmetric and asymmetric schemes.
- Per-tensor and per-channel granularity (configurable channel axis).
- Configurable bit widths (2–16), signed and unsigned, optional narrow range.
- Explicit rounding, clamping, and saturation semantics.
- Power-of-two scale approximation with documented rounding policy.
- Typed quantization-parameter metadata (scale, zero point, qmin, qmax,
  bit width, signedness, scheme, granularity, channel axis).
- Numerically safe handling of constant, all-zero, narrow-range, extreme,
  NaN, and infinite inputs. Errors are raised, never swallowed.

### 3. Custom observers

At minimum:

- `PercentileClippingObserver` — clips calibration range to configurable
  percentiles to resist outliers.
- `PowerOfTwoScaleObserver` — constrains scales to powers of two
  (shift-friendly hardware).
- `MSEGridSearchObserver` — grid-searches clipping thresholds minimizing
  quantization MSE.

Observers must be comparable against PyTorch built-ins on identical data.

### 4. PTQ workflow

- Calibration-based post-training quantization using the stable PyTorch
  quantization APIs available in the installed version (FX graph mode
  preferred; eager mode where FX is impractical).
- Configurable observer/scheme/bit-width per run via typed configs.
- Quick mode must run end-to-end on CPU in minutes.

### 5. QAT workflow

- Fake-quantization-based quantization-aware training for the same models.
- Comparable metrics and artifacts to PTQ runs.

### 6. Layer-level numerical debugging

- Capture per-layer FP32 vs quantized outputs on identical inputs.
- Metrics per layer: MSE, SQNR (dB), cosine similarity, max absolute error,
  and saturation/clipping rates.
- Persist metrics as JSON/CSV artifacts for reporting.

### 7. Sensitivity analysis

- **Proxy sensitivity**: rank layers cheaply from per-layer error metrics
  (no retraining, no per-layer evaluation runs).
- **Ablation sensitivity**: quantize one layer (or group) at a time, measure
  end-task accuracy delta, and rank layers by measured impact.
- Ranked outputs must state which method produced them.

### 8. Hardware-profile abstraction and analytical cost model

- Hardware profiles are data (YAML), validated by typed models: supported
  precisions, per-precision relative compute throughput, memory bandwidth,
  on-chip buffer size, per-layer-type support flags.
- Analytical cost model estimates per-layer and total cost (latency proxy,
  memory traffic, model size) for a given precision assignment.
- All outputs are **estimated** values; the default profile
  (`configs/hardware/generic_edge_npu.yaml`) is fictional.

### 9. Mixed INT4/INT8 precision search

- Search per-layer precision assignments (INT4/INT8, FP32 fallback where a
  layer is unsupported) under hardware-profile constraints (e.g. model-size
  or estimated-latency budget) maximizing predicted accuracy retention using
  sensitivity rankings.
- At least one baseline strategy (greedy) plus one refinement; document the
  algorithm.

### 10. Numerical-regression comparison

- Compare two runs' layer metrics and end metrics; flag regressions beyond
  configurable tolerances with actionable output (which layer, which metric,
  by how much).
- Usable as a CI-style gate returning nonzero on regression.

### 11. CLI

Typer-based `quantscope` command with subcommands covering: environment
inspection, FP32 training/eval, PTQ, QAT, layer diagnosis, sensitivity,
cost estimation, mixed-precision search, run comparison/regression check,
and report generation. `--help` for every command; quick-mode flags.

### 12. Reporting and visualization

- Generate a report (Markdown, optionally HTML) from real run artifacts:
  accuracy tables, per-layer error charts, sensitivity rankings,
  mixed-precision recommendation, cost estimates.
- Every table/chart labels values as measured, simulated, or estimated.

## Result-labeling policy

Every persisted or reported metric is labeled one of:

- **Measured** — actually executed (e.g. FP32/INT8 CPU accuracy, CPU latency,
  serialized file size).
- **Simulated** — numerically emulated (e.g. INT4 fake-quant behavior).
- **Estimated** — analytically derived (e.g. hardware cost model outputs).

CPU latency is never presented as accelerator latency. Simulated INT4 is
never presented as INT4 runtime acceleration.

## Testing requirements

- Core tests: CPU-only, deterministic, no network, no large downloads,
  small synthetic tensors/tiny networks, fast.
- Slow/integration tests marked separately (`-m slow` / `-m integration`).
- Numerical edge cases covered: constant tensors, all-zero tensors, narrow
  ranges, outliers, saturation boundaries, invalid bit widths, NaN/infinity,
  empty calibration data, unsupported operators.
- Quality gates before any phase is declared complete:
  `python -m pytest`, `ruff check .`, `ruff format --check .`.

## Reproducibility requirements

Every experiment records: resolved config, seed, Python version, key package
versions, git commit (when available), device info, dataset info, checkpoint
paths, metrics (JSON/CSV), and the measured/simulated/estimated label for
each result. Datasets, weights, run directories, and large artifacts are not
committed.

## Definition of done

The repository is presentation-ready only when:

- The package installs and the CLI starts successfully.
- Core tests pass; ruff passes.
- Quick mode runs end-to-end.
- An FP32 run and at least two PTQ runs can be compared.
- Layer-level numerical metrics and sensitivity rankings are produced.
- A hardware profile validates; mixed-precision recommendations generate.
- Numerical-regression checks work.
- A report generates from real artifacts.
- The README matches actual behavior.
- Simulated/estimated values are labeled honestly; no number is fabricated.

## Disclaimer

QuantScope does **not** use, emulate, or claim compatibility with proprietary
Quadric technology or any Quadric GPNPU. The bundled hardware profile is
fictional and illustrative. No result in this repository is a measurement
from Quadric hardware or any real NPU.
