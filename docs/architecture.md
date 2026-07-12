# QuantScope Architecture

## Environment (inspected 2026-07-11)

- Platform: macOS (Intel x86_64), CPU-only. MPS not available.
- Python 3.11.13 (conda env `quantscope` at
  `~/opt/anaconda3/envs/quantscope`).
- torch 2.2.2, torchvision 0.17.2 — the last releases publishing
  Intel-macOS wheels. numpy pinned `<2` (ADR-005).
- Quantized engines available: `qnnpack`, `onednn`, `x86`, `fbgemm`.
- `torch.ao.quantization`: available (eager + FX graph mode).
- PT2E (`quantize_pt2e`) and `torch.export`: importable, but immature in
  torch 2.2 — not selected (ADR-006).

## Layering

```
┌─────────────────────────────────────────────────────────┐
│ cli.py (Typer)                                          │
├─────────────────────────────────────────────────────────┤
│ workflows: evaluation/ ptq/qat (quantization/) search/  │
│            sensitivity/ analysis/ reporting/            │
├──────────────────────────────┬──────────────────────────┤
│ torch integration (adapters) │ hardware/ (profiles +    │
│ observers/ models/ data/     │ analytical cost model)   │
├──────────────────────────────┴──────────────────────────┤
│ numerical core: quantization/affine.py, analysis        │
│ metrics — backend-independent (numpy only)              │
├─────────────────────────────────────────────────────────┤
│ config/ (pydantic models)  utilities/ (logging, seeds,  │
│ artifacts, provenance labels)                           │
└─────────────────────────────────────────────────────────┘
```

Principles:

- The numerical core (`quantization/affine.py`, error metrics) depends only
  on numpy — no torch imports — so it is testable and reusable regardless of
  the framework integration API.
- PyTorch-specific quantization lives behind adapters; swapping FX for PT2E
  later touches the adapter layer only.
- Hardware profiles are data (YAML) validated by pydantic models; the cost
  model consumes profiles and layer descriptions, never framework objects.
- Every artifact carries a provenance label: measured / simulated / estimated.

## Package map

| Package | Responsibility |
| --- | --- |
| `quantization/` | Affine math core; PTQ/QAT torch adapters |
| `observers/` | Custom calibration observers |
| `analysis/` | Layer-output capture, numerical-error metrics, regression triage |
| `sensitivity/` | Proxy and ablation sensitivity ranking |
| `hardware/` | Profile schema/loader, analytical cost model |
| `search/` | Mixed-precision search strategies |
| `models/`, `data/` | Tiny CPU-friendly models and datasets |
| `evaluation/` | FP32 train/eval loops, metric collection |
| `config/` | Typed (pydantic) experiment/hardware configs |
| `reporting/` | Report and visualization generation from artifacts |
| `utilities/` | Structured logging, seeding, artifact I/O, env capture |

## Key decisions

See `docs/DECISIONS.md`. Highlights: src layout (ADR-001), framework-independent
core (ADR-002), fictional hardware profile (ADR-003), metric provenance
(ADR-004), numpy<2 pin (ADR-005), FX graph mode as the PyTorch integration
API (ADR-006).
