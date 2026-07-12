# Architectural Decisions

## ADR-001: Source package layout

Use `src/quantscope` so tests import the installed package rather than the repository directory.

## ADR-002: Framework-independent numerical core

Keep quantization arithmetic and metrics independent from PyTorch-specific quantization integration.

## ADR-003: Fictional hardware profile

Use an illustrative hardware profile and never claim it represents proprietary Quadric hardware.

## ADR-004: Metric provenance

Label results as measured, simulated, or estimated.

## ADR-005: Pin numpy < 2

torch 2.2.2 is the last release with Intel-macOS wheels and is compiled
against NumPy 1.x. Under NumPy 2.4, `tensor.numpy()` / `torch.from_numpy`
fail (`RuntimeError: Numpy is not available`). Verified 2026-07-11 in the
`quantscope` env. `numpy>=1.26,<2` is pinned in `pyproject.toml`; interop
re-verified against numpy 1.26.4 with pandas 3.0.3 / scipy 1.17.1 /
matplotlib 3.11.0 importing cleanly. Revisit if the project moves to a
platform with current torch wheels.

## ADR-006: FX graph mode as the PyTorch integration API

Environment inspection (torch 2.2.2) shows eager, FX graph mode, and PT2E
all importable, with quantized engines `qnnpack`/`onednn`/`x86`/`fbgemm`.
PT2E was still maturing in the 2.2 series, so FX graph mode
(`torch.ao.quantization.quantize_fx`) is the primary PTQ/QAT integration,
with eager mode as fallback for modules FX cannot trace. The standalone
numerical core stays numpy-only and backend-independent (ADR-002), and all
torch-specific code sits behind adapters so a future PT2E migration is
localized. `fbgemm`/`x86` is the default engine for INT8 CPU execution on
this machine; INT4 remains simulation-only (no real INT4 backend exists
here — see honesty rules).
