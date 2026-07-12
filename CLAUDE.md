# QuantScope — Claude Code Instructions

## Project purpose

QuantScope is a hardware-aware neural-network quantization, sensitivity-analysis,
and numerical-debugging toolkit for edge accelerators.

This repository is being built as a portfolio project for a new-graduate model
optimization role. It must demonstrate:

- Post-training quantization
- Quantization-aware training
- Calibration and custom observers
- Per-layer numerical-error analysis
- Quantization sensitivity analysis
- Mixed-precision search
- Hardware-aware analytical cost modeling
- Numerical-regression testing
- Reproducible machine-learning experiments

Read `docs/PROJECT_SPEC.md` completely before modifying the repository.

## Required working documents

Keep these files current throughout development:

- `docs/IMPLEMENTATION_PLAN.md`
- `docs/PROGRESS.md`
- `docs/DECISIONS.md`
- `docs/architecture.md`

Update `docs/PROGRESS.md` after every meaningful development stage.

Record important architectural decisions in `docs/DECISIONS.md`.

## Engineering rules

- Use Python 3.11 or newer.
- Use the existing `src/` package layout.
- Use type hints for public functions and methods.
- Write docstrings for public APIs.
- Keep modules focused and reasonably small.
- Use typed configuration models.
- Use structured logging instead of scattered print statements.
- Raise actionable errors rather than silently skipping failures.
- Avoid unnecessary global state.
- Do not place the main implementation in notebooks.
- Do not leave placeholder implementations in required features.
- Do not claim a feature works unless it has been executed or tested.
- Prefer a smaller correct implementation over a broad broken one.
- Do not stop after scaffolding or planning.
- Keep framework-specific quantization code behind adapters where practical.

## Quantization requirements

The project must include:

- Standalone affine quantization mathematics
- Symmetric and asymmetric quantization
- Per-tensor and per-channel quantization
- Configurable integer bit widths
- Saturation and clipping
- Power-of-two scale approximation
- PTQ
- QAT
- At least two custom observers
- Layer-level error metrics
- Sensitivity analysis
- Hardware-constrained mixed-precision search

Prefer the stable PyTorch quantization APIs available in the current environment.
Inspect the installed PyTorch version before selecting the integration API.

Do not silently rely on deprecated APIs. If compatibility code is required,
document it.

## Numerical correctness

Test numerical code carefully.

Important edge cases include:

- Constant tensors
- All-zero tensors
- Narrow ranges
- Outliers
- Integer saturation boundaries
- Invalid bit widths
- NaN and infinity handling
- Empty calibration data
- Unsupported quantized operators

Never swallow numerical errors silently.

## Hardware honesty

The default hardware profile is fictional and illustrative.

Never claim:

- Compatibility with Quadric hardware
- Measurements from a Quadric GPNPU
- Real NPU performance
- Real INT4 acceleration without an actual INT4 backend
- That measured CPU latency represents accelerator latency

Always distinguish:

- Measured model accuracy
- Measured CPU latency
- Exact serialized model size
- Analytical hardware-cost estimates
- Simulated low-precision behavior

Every stored metric should be labeled or documented as one of:

- Measured
- Simulated
- Estimated

## Testing rules

After every meaningful implementation step, run relevant tests.

Before declaring a phase complete, run:

```bash
python -m pytest
ruff check .
ruff format --check .
```

Core tests must:

- Run on CPU
- Avoid large dataset downloads
- Avoid requiring network access
- Use small synthetic tensors or tiny networks
- Complete quickly

Mark slow experiments and integration tests separately.

Do not declare a phase complete while required tests fail.

## Reproducibility requirements

Every experiment should record:

- Resolved configuration
- Random seed
- Python version
- Relevant package versions
- Git commit when available
- Device information
- Dataset information
- Checkpoint paths
- Metrics in JSON or CSV
- Whether each result was measured, simulated, or estimated

Do not commit:

- Datasets
- Model weights
- Run directories
- Large generated reports
- Build artifacts
- Virtual environments

## Development order

Follow this order unless a documented technical reason requires changing it:

1. Inspect the Python, PyTorch, torchvision, and quantization environment.
2. Design configuration schemas and experiment artifact schemas.
3. Implement affine quantization arithmetic.
4. Implement custom observers.
5. Implement numerical-error metrics.
6. Implement FP32 training and evaluation.
7. Implement PTQ.
8. Implement layer-output capture and numerical diagnosis.
9. Implement sensitivity analysis.
10. Implement the hardware profile and analytical cost model.
11. Implement mixed-precision search.
12. Implement QAT.
13. Implement numerical-regression comparison.
14. Implement reports and visualizations.
15. Complete documentation, CI, and repository cleanup.

## Implementation behavior

- Work autonomously and make reasonable technical decisions.
- Ask a question only when genuinely blocked.
- Run tests after each meaningful component.
- Fix failures before moving to the next phase.
- Keep `docs/PROGRESS.md` current.
- Record important decisions in `docs/DECISIONS.md`.
- Use CPU-compatible synthetic tests during development.
- Avoid large downloads unless they are necessary and documented.
- Preserve useful experiment logs and failure information.
- Prefer a working vertical slice over many incomplete features.

## Definition of done

The repository is ready for presentation only when:

- The package installs successfully.
- The CLI starts successfully.
- Core tests pass.
- Quick mode runs end to end.
- An FP32 run and at least two PTQ runs can be compared.
- Layer-level numerical metrics are generated.
- Sensitivity rankings are produced.
- A hardware profile is validated.
- Mixed-precision recommendations are generated.
- Numerical-regression checks work.
- A report is generated from real artifacts.
- The README matches actual behavior.
- Simulated and estimated values are labeled honestly.
- No benchmark number is fabricated.
