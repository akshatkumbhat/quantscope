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

## ADR-007: Bounded deterministic sampling for distribution observers

Percentile and MSE-grid-search observers need the value distribution, not
just min/max. Instead of unbounded storage or complex streaming histograms,
each batch is subsampled to at most `samples_per_batch` (default 8192)
elements with a fixed-seed RNG. Memory grows linearly with calibration
batch count (small by design), results are reproducible across runs, and
the implementation stays simple enough to test exhaustively. Per-channel
granularity is explicitly rejected for these two observers (out of scope);
`MinMaxObserver` and `PowerOfTwoScaleObserver` support it. The
`PowerOfTwoScaleObserver` rounds scales **up** to the next power of two so
snapping never introduces clipping (up to 2x coarser steps instead), and
recomputes the zero point for asymmetric schemes so zero stays exactly
representable.

## ADR-008: Texture-10 benchmark A and simulation policy v1

The original synthetic task saturated (FP32 = INT8 = 1.0 accuracy), so the
benchmark measured nothing. Following an external design review, benchmark
A uses: (1) **Texture-10** — 10 sinusoid-texture classes with deliberately
small parameter separations, 15–25% boundary examples interpolated 30–45%
toward a neighboring prototype (label preserved), SNR-controlled noise, and
analytic rotation/translation (orientation/phase jitter, no warping
artifacts); (2) a ~20k-param **BottleneckResNet** whose structure (residual
bypasses vs. a 6-channel bottleneck vs. an unprotected downsample) gives
per-layer sensitivity a reason to be non-uniform; (3) **NLL and correct-class
margin** as primary discrimination metrics, because top-1 accuracy is
discrete and produces false ties.

**Simulation policy v1** (`quantization/simulate.py`): weights per-channel
symmetric at W bits; model input + ReLU outputs per-tensor asymmetric at A
bits, min-max calibrated; BN unfolded, logits unquantized. All results from
this path are labeled **simulated**. It intentionally does not match
backend INT8 semantics — the backend-matched profile is deliverable C.
Boundary-example interpolation was chosen over label noise because label
noise lowers the ceiling without making the decision function more
quantization-sensitive. Gates (dev seed first, then 3 frozen seeds): FP32
88–94%, W4A4 clearly degraded in NLL/margin/accuracy, W8A8 drop nonzero
but smaller; max 3 tuning iterations, then stop and report.

### ADR-008 addendum: frozen recipe and accepted gate deviation (2026-07-11)

Tuning findings over the 3-iteration budget (dev seed 0):

- Iter 1 (boundary 0.20, λ∈[0.30,0.45], SNR 8 dB): FP32 99.95% — saturated.
- Iter 2 (boundary 0.35, SNR 4 dB): FP32 99.95% — unchanged. Boundary
  fraction and SNR alone do not create error here.
- Iter 3 (boundary 0.45, λ∈[0.40,0.50], SNR 4 dB): FP32 96.0%,
  W8A8 95.95%, W4A4 91.6%; NLL 0.102 / 0.103 / 0.211; margin
  5.91 / 5.94 / 4.79.

Diagnosis: label-preserving interpolation below λ=0.5 leaves the true
class with a detectable energy advantage, so a converged model incurs no
irreducible error; only interpolation *near* λ=0.5 produces real
ambiguity. Boundary fraction and SNR are weak difficulty levers for this
model class.

Decision (user-approved): freeze the iteration-3 recipe as the benchmark-A
default; close A as a **conditional pass with a documented deviation** —
FP32 misses the 88–94% band high by ~2 pp. The band was not widened after
seeing results. Acceptance over 3 fixed seeds requires: aggregate
NLL/margin monotone FP32 > W8A8 > W4A4; mean W4A4 accuracy clearly
degraded; W8A8 less degraded than W4A4; mean FP32 ≤ 96% with no seed
saturated; W8A8 effect required in NLL/margin only (accuracy is discrete).
No generator retuning before plan step B; if per-group ablation shows 96%
obstructs sensitivity analysis, class-separation parameters get revisited
as a separately approved change
(`scripts/check_texture_a_acceptance.py` encodes the conditions).

### ADR-008 addendum 2: 3-seed validation result — A closed as PARTIAL PASS

Checker result (conditions unchanged, recorded as agreed; script exits 1
on the frozen recipe and that is the accepted record):

- Mean FP32 accuracy 96.22% — **failed** the ≤96% condition by 0.22 pp.
- W8A8 mean-margin direction — **failed on all three seeds** (margin rose:
  5.905→5.935, 6.425→6.432, 5.765→5.784).
- W8A8 NLL effect — passed (aggregate 0.0938 → 0.0938, +4e-5).
- W4A4 degradation and bit-width discrimination — passed (mean accuracy
  −2.2 pp, NLL +64%, margin −0.69; degradation present in every seed).
- Tuning budget — exhausted (3 iterations).

Disposition (user decision, 2026-07-12): **partial pass with two
documented deviations; benchmark usable pending the B sensitivity gate.**
The FP32 cap was not raised and the margin condition was not removed
post hoc. The 0.22 pp cap miss alone does not justify another tuning
cycle.

Prospective interpretation for future phases: mean correct-class margin
aggregates differently from NLL (it is dominated by well-classified
samples and can move opposite to NLL for near-zero effects), so it must
not be used as a guaranteed-monotone acceptance metric for very small
W8A8-scale effects. It remains a valid *diagnostic* and a valid
discrimination metric for large effects (W4A4 moved it −0.69).

**B stop gate:** if per-group W4A4 ablations show meaningful,
non-uniform sensitivity in ΔNLL, prediction flips, or accuracy — continue
on the frozen recipe. If the ranking is effectively flat, tie-dominated,
or unstable across seeds — stop before C and open the generator-level
class-separation change as a new approved tuning phase.

### ADR-008 addendum 3: B stop gate result — STOP (2026-07-12)

Per-group W4A4 ablation over the 3 seed checkpoints
(`scripts/check_sensitivity_gate.py`, thresholds fixed before running):

- Criterion 1 (meaningful) — **passed**: max mean ΔNLL +0.014 (block_b),
  flip rates up to 2.9%.
- Criterion 2 (non-uniform) — **passed**: max/median mean-ΔNLL ratio 3.76;
  block_b and stem lead, block_a groups near zero (block_a_conv2 slightly
  negative), consistent with the residual-bypass design intent.
- Criterion 3 (stability) — **FAILED**: pairwise Spearman 0.405 / 0.405 /
  0.095 (0 of 3 pairs ≥ 0.7); mean top-2 {block_b, stem} reproduced in
  only 1 of 3 individual seeds. Per-seed ΔNLL for block_b spans +0.002 to
  +0.024; bottleneck is negative on seed 2.

Interpretation: single-group W4A4 effects (ΔNLL ≤ 0.024 on a ~0.09
baseline) are the same order as inter-seed variance — the task is still
too easy for one group's quantization noise to reliably move the loss.
This is the failure mode the gate was designed to catch.

Disposition: **stopped before plan step C.** The generator-level
class-separation change (tighter inter-class parameter separations in
`texture10._class_components`) is proposed as a new, separately approved
tuning phase; B3 (exhaustive mixed-precision search) is deferred until a
stable ranking exists, since Pareto/search comparisons against a noise-
dominated sensitivity signal would be meaningless.

## ADR-009: Generator class-separation phase (approved 2026-07-12)

Scope controls (user-set):

- Change **only** the primary-component frequency separation in
  `texture10._class_components` (`freq_step`, previously hard-coded 0.30).
  Model, training recipe, boundary interpolation, noise, quantization
  settings, the eight group definitions, and the B2 analysis are
  unchanged.
- **Intent statement:** the freq-step change reduces class margin; it is
  NOT intended to manufacture a particular layer ranking. Any resulting
  ranking — including a change from block_b/stem leadership — is valid if
  meaningful, heterogeneous, and stable.
- Development seed 7 (never seeds 0/1/2) selects the candidate.
- Predeclared candidates, in order: **0.20, 0.15, 0.12.** Stop at the
  first whose dev-seed FP32 lands near the middle of the target band
  (preferably ~90–93%) with clear FP32 > W8A8 > W4A4 discrimination.
  A single-seed ablation may be used as a screening diagnostic only;
  Spearman stability is not a dev gate (one seed cannot establish it).
- After selection: freeze, run the unchanged 3-seed validation
  (seeds 0/1/2), and re-run the full B2 gate without amending criteria:
  FP32 mean in the original 88–94% band; meaningful effect; non-uniform;
  Spearman ≥ 0.7 in ≥ 2 of 3 pairs; existing top-group reproducibility.
- Candidates are never chosen using the 3-seed validation. If the frozen
  candidate fails validation, the phase is recorded **failed** and work
  stops for reassessment — no fourth tuning cycle.

### ADR-009 addendum: phase FAILED at candidate screening (2026-07-12)

Dev-seed-7 results (FP32 / W8A8 / W4A4 accuracy; NLL in parens):

| freq_step | FP32 | W8A8 | W4A4 |
| --- | --- | --- | --- |
| 0.30 baseline (seeds 0–2 mean) | 96.2% | 96.1% | 94.0% |
| 0.20 | 96.8% (0.090) | 96.6% (0.091) | 93.5% (0.173) |
| 0.15 | 96.5% (0.096) | 96.4% (0.096) | 93.7% (0.162) |
| 0.12 | 96.0% (0.106) | 96.1% (0.107) | 90.9% (0.211) |

No candidate approached the 88–94% band (target ~90–93%); halving the
frequency separation moved FP32 by < 1 pp. **Diagnosis:** the 18°
orientation step between adjacent classes carries most of the class
identity, so compressing the frequency axis alone cannot collapse class
margins. (Secondary observations: W4A4 damage *grew* with smaller steps —
NLL +0.105 at 0.12 vs +0.077 at 0.20 — and the 0.12 run shows another
small-effect W8A8 accuracy inversion, +0.1 pp, consistent with the
ADR-008 margin finding.)

Disposition: per the pre-agreed protocol, no candidate is frozen, the
3-seed validation was not run, and no fourth tuning cycle was started.
Reassessment options (user decision pending): (a) a new scoped generator
change targeting the orientation step — the demonstrated dominant axis;
(b) accept FP32 ~96% and test whether ranking *stability* (the actual B
blocker) improves anyway, since per-group W4A4 effects grew at smaller
freq_step; (c) a different difficulty mechanism entirely.
