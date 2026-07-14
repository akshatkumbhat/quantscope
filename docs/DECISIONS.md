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

### ADR-009 addendum 2: option-(b) diagnostic — predeclared interpretation

The frequency-separation phase remains FAILED against its original FP32
target; this diagnostic answers a *new* question on the untouched dev
seed 7: did freq_step=0.12 create enough per-group quantization signal
to justify testing stability directly?

Predeclared (2026-07-13, before seeing the result):

- **Promising** only if the strongest group's ΔNLL ≥ 0.028 (~2x the
  previous 3-seed max mean of 0.014), AND the top group exceeds the
  median group by ≥ 0.015, AND prediction flips / accuracy move
  consistently with the larger NLL effect.
- One seed cannot test stability; the diagnostic's rank order is not
  interpreted.
- If promising: open a prospectively re-scoped validation phase for
  freq_step=0.12 — recording BEFORE touching seeds 0–2 that FP32 88–94%
  is a desired property rather than the hard gate, and the decisive gate
  is 3-seed sensitivity stability — then train and ablate seeds 0–2
  unchanged.
- Otherwise (≤ ~0.024 scale, flat, or driven by one anomalous group):
  reject option (b) and proceed to the scoped orientation-step
  experiment.

### ADR-009 addendum 3: diagnostic PROMISING — re-scoped validation opened

Diagnostic result (dev seed 7, freq_step=0.12, W4A4 per group): block_b
ΔNLL +0.0746 (flips 6.0%, accuracy −3.35 pp); stem +0.0108; median
+0.004. Predeclared conditions: top ≥ 0.028 ✓; top − median ≥ 0.015 ✓
(+0.071); flips/accuracy consistent ✓.

Judgment call on the "one anomalous group" rejection clause, recorded
with reasons: block_b's dominance is ruled NOT anomalous because (1) it
was already the top group in the 3-seed ablation at freq_step 0.30, so
this strengthens an existing pattern, and (2) it is the only group with
two conv layers and two activation sites — double the quantization
surface. Caveat: groups ranked 3–8 remain at the old noise scale, so
full-ranking Spearman may stay weak even if top-group leadership is
reproducible.

**Pre-registration for the re-scoped validation phase (recorded before
any seed-0/1/2 run):** FP32 88–94% is now a *desired property*, not the
hard gate. The decisive gate is **3-seed sensitivity stability**
(unchanged B2 criteria: meaningful, non-uniform, Spearman ≥ 0.7 in ≥ 2
of 3 pairs OR existing top-group reproducibility). The
frequency-separation phase's original FAILED verdict against its FP32
target stands unamended. Plan: train seeds 0/1/2 at freq_step=0.12 with
the otherwise-frozen recipe, ablate each, run the unchanged
check_sensitivity_gate.py.

### ADR-009 addendum 4: re-scoped validation FAILED the decisive gate

3-seed results at freq_step=0.12 (unchanged recipe, unchanged checker):

- FP32 95.7 / 95.4 / 95.0 (mean 95.4%) — desired band still missed
  (recorded as observation; not the gate).
- Discrimination intact: W4A4 clearly degraded on every seed; W8A8
  accuracy inversions of +0.05–0.15 pp on all three seeds (consistent
  with the known small-effect behavior).
- **Stability gate FAILED, worse than at 0.30**: Spearman 0.571 / −0.452
  / 0.071 (0 of 3 pairs ≥ 0.7); top-2 reproduced 1/3. block_b per-seed
  ΔNLL: +0.0484 / +0.0239 / **−0.0006** — the dev-seed-7 signal
  (+0.0746) did not replicate; on seed 2 the effect vanished.

Interpretation: the diagnostic's promise was seed luck. Per-group W4A4
sensitivity at this model scale is dominated by which weight
configuration a given training run happens to reach, not by
architecture: the same group swings from the strongest effect to zero
across seeds. Larger mean effects did not stabilize the ranking.

Disposition: option (b) is exhausted; per pre-registration this phase is
**FAILED** and work stops for reassessment. Candidate reassessment
directions (user decision pending): (a) the scoped orientation-step
generator change (still untried; lowers FP32 into band and raises all
effect sizes, but freq-step evidence shows bigger effects do not
automatically stabilize rankings); (b2) reframe the deliverable —
accept that sensitivity rankings are checkpoint-specific at this scale,
document that as a finding, and run B3's exhaustive search
per-checkpoint (search-vs-optimal regret is well-defined per model and
does not require cross-seed rank stability); (c) a cheap predeclared
diagnostic first: rerun the ablation at a harsher target (e.g. W3A3) on
the three existing freq_step=0.12 checkpoints — no retraining — to test
whether instability is effect-size-limited before any further generator
work.

## ADR-010: B3 mixed-precision search — reframed deliverable (2026-07-13)

Frozen conclusions carried forward unchanged (user decision): B2 failed
its preregistered cross-seed stability gate; the failure reproduced under
two generator settings; increasing the mean effect did not stabilize the
ranking; therefore **W4A4 layer sensitivity is checkpoint-conditioned for
this benchmark and model scale and must not be presented as an
architecture-level property.** This is a result, not a defect to tune
away. W3A3 is retained only as an optional later stress-test appendix,
not as a rescue of the failed criterion.

### B3 pre-registration (recorded before any sweep ran)

Inputs: the three validated freq_step=0.12 checkpoints (seeds 0/1/2), no
seed selection; per-checkpoint sensitivity = that checkpoint's existing
W4A4 ablation ΔNLL ranking.

Per checkpoint: exhaustively evaluate all 2^8 = 256 INT4/INT8 group
assignments (each group at W4A4 or W8A8, simulation policy v1);
construct the exact NLL-vs-cost and accuracy-vs-cost Pareto frontiers;
evaluate sensitivity-ranked and greedy searches (simulated on the
exhaustive table); compare to random search and exact optima; transfer
each seed's ranking to the other two checkpoints and measure regret.

Cost model (labeled **estimated**): normalized weight-storage bits —
cost(config) = Σ_g params(g)·bits(g) / Σ_g params(g)·8; all-INT8 = 1.0,
all-INT4 = 0.5. Activation bits follow the group but do not enter cost
(documented simplification; the analytical hardware cost model is a
later deliverable).

Predeclared primary metrics:

1. **Regret at fixed budget**: NLL(best found with cost ≤ 0.75) −
   NLL(exact best with cost ≤ 0.75); accuracy regret reported alongside.
2. **Evaluations to frontier**: number of table evaluations until
   best-found budget-regret ≤ δ = 0.01 NLL.
3. **Pareto overlap across checkpoints**: Jaccard similarity of the
   Pareto-optimal assignment sets, per seed pair.
4. **Cross-seed rank-transfer regret**: seed A's sensitivity ordering
   applied as an INT4-flip path on seed B's table; regret at budget 0.75
   for all 6 ordered pairs.
5. **Random-search distribution**: 10 deterministic search seeds × 32
   uniform samples each; distribution of budget-regret.

Search definitions: sensitivity path = start all-INT8, flip groups to
INT4 in ascending ΔNLL order (9 path points; ranking cost = the 8 prior
ablation evals, footnoted); greedy = start all-INT8, commit the
lowest-NLL-increase flip each round (36 evaluations); both simulated
against the exhaustive table.

Success criteria: the sweep produces a nontrivial precision/quality
tradeoff, and the search methods are evaluable against exact optima.
Explicitly NOT required: shared rankings or shared Pareto frontiers
across checkpoints. Cross-checkpoint transfer is measured, never
assumed.

### ADR-010 addendum: B3 results (2026-07-13)

Methodological success criteria: **met.** Nontrivial tradeoff (budget-
feasible NLL spreads 0.029–0.062, all » δ=0.01; frontiers of 15–18
points, 14–16 of them mixed-precision) and every search evaluated
against exact optima.

Substantive results, reported per the two predeclared questions:

**Q1 — within-checkpoint utility: NEGATIVE.** The ablation-ranked
single path (flip groups to INT4 in ascending ΔNLL order) had budget
regret 0.0465 / 0.0338 / 0.0036 across seeds and never came within δ of
the frontier on seeds 0–1. Greedy search (36 evals, measures joint
effects incrementally) reached regret 0.0095 / 0.0025 / 0.0009 in
21 / 6 / 6 evaluations. Random search (32 evals) had median regret
0.0035 / 0.0018 / 0.0011 — **beating the sensitivity path on every seed
and matching or beating greedy.** The hoped-for finding ("a ranking
measured on a checkpoint still guides search for that checkpoint") is
NOT supported: one-at-a-time W4A4 ablation effects do not compose
additively into joint mixed-precision quality; interaction effects
dominate. Caveat recorded: 32 random samples cover 12.5% of this
256-point space — random search would not scale this way to larger
spaces; the predeclared design nevertheless stands as run.

**Q2 — cross-checkpoint transfer: erratic, consistent with weakly
informative rankings.** Transfer penalties ranged from +0.0222 to
−0.0411; seed 2's ranking outperformed seeds 0 and 1's *own* rankings
on their own tables — a foreign ranking beating the native one is
further evidence the rankings are noise-dominated. Pareto-frontier
overlap across checkpoints is small (Jaccard 0.065–0.161): the optimal
mixed-precision assignments are also checkpoint-specific.

Honest summary for the report/README: at this model scale, exhaustive
enumeration shows (a) mixed-precision tradeoffs are real and per-
checkpoint optima exist; (b) greedy joint-effect search works; (c)
one-shot ablation sensitivity rankings are NOT a reliable guide even on
the checkpoint they were measured on, and neither rankings nor Pareto
sets transfer across training runs. All quality metrics simulated;
costs estimated; no claim generalizes beyond this benchmark and scale.

## ADR-011: Plan step C — graph-anchored backend parity (2026-07-13)

C is an independent numerical-validity phase; it does not support the
failed sensitivity heuristic (B conclusions frozen as recorded).

**Comparison ladder:** `sim_backend_matched` ↔ `reference_fx` ↔
`real_int8`. `sim_custom` (policy v1) stays out of the parity gate
until D. One frozen checkpoint (seed 0, freq_step=0.12) first; expand
only if it passes and reruns are cheap.

**Construction (user-selected option B): the graph-anchored
backend-matched simulator** — hold Torch's fusion, placement,
calibration statistics, and graph topology constant while replacing
Torch's affine quantize/dequantize arithmetic with QuantScope's.
Activations: deep-copy the calibrated prepare_fx model, freeze each
activation observer's searched range, and swap the observer for a
non-updating `FrozenQuantScopeFakeQuant` powered by the affine core.
Weights: instantiate the configured Torch weight observer per fused
weighted module (matching conversion behavior), extract per-channel
ranges, fake-quantize the folded weight with QuantScope arithmetic,
bias stays FP32, fused structure preserved. Torch's `FakeQuantize`
module is NOT used in the primary comparison (it would validate Torch
against Torch).

**Inspected Torch 2.2.2 defaults (measured, not assumed):** engine
fbgemm; activations HistogramObserver, quint8, per-tensor affine,
quant range [0, 127] (reduce_range=True); weights
PerChannelMinMaxObserver, qint8, per-channel symmetric, [-128, 127],
ch_axis 0; 14 activation observer sites on the fused BottleneckResNet
graph; BN fully folded (ConvReLU2d fusions). Confirmed: Torch 2.2.2
symmetric scale = max_abs / ((quant_max − quant_min)/2) — denominator
127.5, vs QuantScope's qmax − zp = 127. QuantScope's general semantics
are NOT silently changed: a `qparam_policy="torch_2_2"` compatibility
mode is added, and the artifact shows both calculations and their
~0.39% systematic difference as a compatibility finding.

`HistogramObserver._non_linear_param_search()` (private, version-
pinned) is contained behind one extractor with a torch-version
assertion and a characterization test (qparams from extracted bounds
must equal `calculate_qparams()`); both raw histogram extent and
searched extent are recorded. Nothing else in QuantScope may touch the
private method.

**Staged comparisons (failures stay localized):** (1) qparam parity at
all 14 activation sites and every weight channel (exact equality, max
abs diff, relative scale diff); (2) primitive fake-quant parity on
captured tensors vs Torch's fake-quant ops with identical frozen
qparams — integer codes and dequantized values, including halfway,
saturation, zero, constant-range, and negative cases; (3) activation-
only model parity; (4) weight-only model parity; (5) full simulation vs
`convert_to_reference_fx`; then reference_fx ↔ convert_fx real INT8.

**Recorded per path:** accuracy, NLL, prediction disagreement rate,
logit MSE/SQNR/cosine/max-abs, per-sample logit differences, scale/
zero-point metadata, quantized-node coverage and float islands, and
prepared/reference/backend graph summaries. Identical checkpoint,
calibration sample IDs and order, eval set, preprocessing, and fused
model across all paths.

**Tiered acceptance:** sim ↔ reference is the strict gate (differences
mean a semantic mismatch to localize); reference ↔ real INT8 tolerates
small numerical differences but must stay materially aligned in
accuracy/NLL/predictions. Any mismatch must be localized with evidence
— never attributed generically to "kernel differences". Success does
not require bit-exact equality or an INT8 accuracy drop. Option A
(independent end-to-end simulator with own fusion/placement) is a
separately-named later test and never shares C's strict gate. W3A3
deferred; no custom Torch observer adapter in C.

### ADR-011 addendum: C results — PASS on all 3 checkpoints (2026-07-13)

**Stage 1 (qparam parity):** exact. Activation scales match to ≤1e-7
relative (float32 representation), all zero points equal; weight scales
bit-exact under `qparam_policy="torch_2_2"` at all 9 sites, zero points
equal. The native-policy comparison shows the predicted uniform
127.5/127 ratio (1.00394) at every weight site — compatibility finding
#1, shown in the artifact, both calculations recorded.

**Stage 2 (primitive parity):** 3–4 code mismatches per ~6.3M captured
elements per seed, each exactly one code, each at a float32 .5
rounding-tie boundary — compatibility finding #2: **Torch quantizes
with float32 division; QuantScope uses float64**, so exact ties resolve
differently (characterized in unit tests; core semantics unchanged by
policy discipline).

**Stages 3–5 + backend:** activation-only and weight-only diagnostics
show 0 (one seed 0.05%) prediction disagreement and 56–65 dB logit
SQNR. Strict gate (sim_full ↔ reference_fx): prediction disagreement
0.15% / 0.15% / 0.50%; logit SQNR ~35 dB; residuals decompose into
**integer multiples of the final logit quantization scale** (~95% of
samples differ by exactly one final code; rare tails to 7 codes from an
upstream tie amplified through subsequent layers; max fractional
deviation from integer 0.0069). Root cause: chained quantized ops emit
lattice-aligned values, making division ties *systematic* at
requantization nodes rather than rare — the same finding #2, amplified
by graph structure. Accuracy/NLL materially aligned (Δaccuracy ≤ 0.1
pp, ΔNLL ≤ 1.2e-3). Backend gate (reference_fx ↔ real_int8):
**0.0000 prediction disagreement on all three seeds**, accuracy/NLL
identical to 4 decimals, SQNR 52–56 dB.

**Graph coverage (recorded per artifact):** reference graph 73 nodes
with 28 quantize/dequantize nodes; float islands = the two residual
adds and flatten (adds execute in float between dq/q pairs, as expected
for this qconfig); lowered INT8 graph 23 nodes, 4 q/dq boundary nodes,
quantized fused modules, only flatten in float. All 9 weighted modules
quantized in both converted paths.

**Verdict:** C succeeds under its predeclared criteria — the three
paths are traceably comparable and every residual difference is
measured, localized, and explained by two named compatibility findings.
Optional future work (not required): a float32-division compat mode for
tie-exact parity, only if D needs it.

## ADR-012: Plan step D — observer-policy comparison study (approved
with amendments, 2026-07-14)

D isolates one causal question: the same frozen checkpoint and the same
clean evaluation examples are quantized using activation ranges derived
from clean versus outlier-contaminated calibration data, with observer
policy as the only changing quantization factor. `sim_custom`
(simulation policy v1) enters for the first time. B's conclusions stay
frozen; C's validated arithmetic is the foundation. No torch observer
adapter; W3A3 still deferred.

### Design (user amendments incorporated)

1. **No retraining.** The three frozen clean-trained freq_step=0.12
   checkpoints serve both conditions; retraining on stressed data would
   change learned representations and confound the calibration
   comparison. Stress applies only to calibration/evaluation inputs.
   Stress-trained checkpoints are at most a later optional appendix.
2. **Paired factorial** (calibration → evaluation): clean→clean,
   stressed→clean, clean→stressed, stressed→stressed. **Primary
   condition: stressed calibration → clean evaluation** — rare
   irrelevant calibration outliers expand MinMax ranges, reducing
   resolution for ordinary clean inputs.
3. **One stress mechanism**: impulses only. 0.2% of pixels; signs
   balanced deterministically; magnitude ±6 per-image standard
   deviations; injected AFTER blur; labels, base sample IDs, and all
   texture parameters preserved (stress is applied to the finished
   clean dataset, so pairing holds by construction). This pairs with
   the frozen 0.1/99.9 percentile observer: ~0.1% of injected mass per
   tail gives the clipping level a prospective mechanism-level
   rationale. Glints are a separately named secondary mechanism, only
   after the impulse study completes.
4. **Frozen observers**: MinMax (baseline); Percentile 0.1/99.9;
   MSE-grid (defaults); PowerOfTwo (round-up). Weights fixed to
   per-channel symmetric MinMax in every arm. No post-hoc tuning; the
   percentile stays 0.1/99.9 (0.5/99.5 would decouple the clipping
   level from the preregistered outlier mass).
5. **Configurations** (notation: WxAy = x-bit weights, y-bit
   activations): W4A4 primary discrimination; **W8A4
   activation-isolation** (observer policy is the independent variable
   while weights stay high precision); W8A8 backend-like secondary.
   W4A4 runs the full 4-condition factorial; W8A4/W8A8 run
   clean→clean and stressed→clean (stressed-evaluation arms added only
   because they are cheap; secondary).
6. **Stress-design gate** (dev seed 7's clean-trained checkpoint,
   before touching validation seeds): (a) stressed calibration expands
   the observed MinMax range by ≥25% at ≥50% of activation sites —
   policy v1 has 9 sites (input + 8 ReLUs), so ≥5 of 9; the amendment's
   "7 of 14" assumed C's FX-graph site count and is adapted
   proportionally, recorded here; (b) MinMax W4A4 NLL on the unchanged
   clean evaluation set worsens by > 0.02 under stressed vs clean
   calibration; (c) labels/sample IDs/non-stress generator values
   verified identical between paired sets. One mechanical fallback:
   impulse magnitude 6σ → 10σ, decided ONLY on range expansion and
   MinMax degradation (never on robust-observer performance). Both
   levels fail ⇒ stress-design failure, stop.
7. **Metrics**: ΔNLL and Δaccuracy vs the checkpoint's FP32 (primary:
   stressed-calib → clean-eval W4A4 ΔNLL); per-site calibrated scales
   and saturation rates (mechanism evidence); per-site activation SQNR
   on a fixed held-out probe batch (seed stream +3, NOT the calibration
   batch, to avoid favoring the observer's fitted distribution),
   evaluated on both the clean probe and its paired stressed version;
   power-of-two scale property verified exactly.
8. **Predeclared interpretation.** Q1 (robustness) confirmed iff, in
   stressed-calib → clean-eval W4A4: percentile or MSE-grid improves
   mean NLL over MinMax by > 0.01; direction favorable in ≥ 2 of 3
   checkpoints; accuracy not worse by > 0.5 pp; and the benefit is not
   driven by catastrophic saturation at a single site. Q2
   (non-inferiority) confirmed iff, in clean→clean, no robust observer
   is worse than MinMax by > 0.005 mean NLL at the same precision
   configuration (never pooled across configurations). Q3
   (power-of-two cost) is measurement-only, reported per configuration
   and per checkpoint. Cross-seed ranking stability is reported, not
   gated (B's lesson). Negative results get equal prominence.
9. **Q4 (optional, non-gating)**: sim_custom-MinMax ↔
   sim_backend_matched at W8A8 with identical calibration inputs;
   differences attributed explicitly to observer range-selection
   policy, graph placement, or arithmetic policy. Must not delay the
   primary study.

### ADR-012 addendum: stress-design gate FAILED on criterion (a) —
stopped before validation seeds (2026-07-14)

Dev-seed-7 results (`scripts/check_stress_gate.py`):

| criterion | 6σ | 10σ (fallback) |
| --- | --- | --- |
| (a) ≥25% MinMax scale expansion at ≥5/9 sites | **4/9 — FAIL** | **4/9 — FAIL** |
| (b) MinMax W4A4 NLL degradation > 0.02 (clean eval) | +0.0920 — pass | +0.6864 — pass |
| (c) pairing identity | pass | pass |

Per-site pattern (10σ): input 3.14×, stem 2.85×, block_a.relu1 2.60×,
block_a.relu_out 2.15×, down_relu 1.23× (just under threshold), all
deeper sites exactly 1.00×.

Localization: the *mechanism* works — stressed calibration destroys
MinMax resolution for clean inputs (criterion (b) exceeded by 4.6× at
6σ and 34× at 10σ). What fails is the **site-coverage expectation**
encoded in criterion (a): isolated impulses are spatially attenuated by
convolution and downsampling, so ranges beyond the first block never
inflate, regardless of magnitude. Raising magnitude amplifies early
sites (already far past threshold) without propagating depth-wise.
Criterion (a) assumed outliers reach most of the network; CNNs
structurally prevent that for pixel impulses.

Disposition: per the pre-registered protocol (both magnitude levels
exhausted), **stress-design failure recorded; D stopped before the
validation seeds.** No threshold was adjusted after seeing results.
Reassessment options (user decision): (i) accept failure and close D;
(ii) prospectively amend criterion (a) with the attenuation rationale
(e.g. require expansion only at the sites where impulses physically
survive) and rerun the gate as a new decision; (iii) switch to the
glint mechanism (larger spatial footprint; plausibly survives
downsampling and inflates deeper sites) as the separately named
secondary stress, promoted with its own gate.
