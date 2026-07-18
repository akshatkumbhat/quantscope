# Implementation Plan

Status: **all phases complete** (v0.1.0, 2026-07-18). This plan is
retained for historical context; the authoritative records are
`docs/PROGRESS.md` (chronology) and `docs/DECISIONS.md` (ADR-001..016
with results). Definition-of-done verification lives in the ADR-015/
016 result addenda (green CI on 3.11/3.12, clean-clone check).

| Phase | Scope | Outcome |
| --- | --- | --- |
| 1 | Environment inspection, config/artifact design | done (ADR-001..006) |
| 2 | Affine quantization core + numerical tests | done |
| 3 | Custom observers + error metrics | done (ADR-007) |
| 4 | FP32 training, calibration, FX PTQ | done |
| 5 | Layer-level numerical debugging | done (parity capture, ADR-011) |
| 6 | Sensitivity analysis | done — honest negative finding (ADR-010) |
| 7 | Hardware profiles, cost model, mixed-precision search | done (ADR-010/014) |
| 8 | QAT | done + confound control (ADR-013/016) |
| 9 | Regression harness, reports, visualizations | done (ADR-015, reporting/) |
| 10 | Documentation, CI, cleanup | done (mypy+ruff+tests+smoke in CI) |
