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

### ADR-012 addendum 2: Impulse Stress Gate v2 (pre-registered
2026-07-14, before the fresh dev run)

Original gate result stands unchanged: the propagation-coverage
criterion failed at both magnitudes; its assumption that isolated
impulses should expand ranges throughout the CNN was **falsified**; no
validation checkpoint was examined. Downstream attenuation is recorded
as a mechanism finding, not a failure: *pixel impulses strongly
contaminate input and early-layer calibration ranges but are attenuated
by convolution and pooling before deeper observer sites; early-layer
range contamination alone is sufficient to damage end-to-end W4A4
performance.* Glints are NOT promoted; they remain a separately
preregistered stress family considered only if v2 fails.

v2 reflects the actual causal requirement — calibration outliers must
materially distort the ranges that physically survive, and must damage
clean-input quantized performance; they need not reach most sites.

- **Fresh evidence base**: a newly trained dev checkpoint (seed 8,
  frozen 0.12 recipe) — not the seed-7 results that motivated this
  amendment. Stress fixed at **6σ** (the +0.092 NLL effect at 6σ was
  already ample; 10σ is unnecessarily extreme).
- **Criterion 1 — early-site reach** (structural definition fixed
  before running): eligible sites are the input plus the activation
  observers before the first spatial downsampling boundary
  (`down_conv`, stride 2) — i.e. `__input__`, `stem_relu`,
  `block_a.relu1`, `block_a.relu_out`. Require ≥3 of these 4 to show a
  MinMax scale increase ≥25%, and the input ≥2×.
- **Criterion 2 — behavioral discrimination**: stressed-calib →
  clean-eval MinMax W4A4 NLL worse by > 0.02 vs clean-calib →
  clean-eval. Accuracy and prediction flips reported, not gated.
- **Criterion 3 — pairing integrity**: labels, sample IDs, base
  textures, non-stress generator parameters exactly equal.
- **Criterion 4 — no observer shopping**: the gate inspects MinMax
  only; percentile/MSE-grid/pow2 results stay hidden until the stress
  design passes.
- **Mechanism decomposition added to the final D study**: starting from
  clean-calibration qparams, substitute stressed qparams one site at a
  time and cumulatively by stage (input / remaining early sites /
  deeper sites), attributing the MinMax NLL damage. If nearly all
  damage is the input observer, the result is reported specifically as
  **input/early-activation calibration robustness**, never
  network-wide observer robustness.
- Pass ⇒ proceed to validation seeds under the existing D comparison.
  Fail ⇒ close the impulse mechanism; consider glints as a separately
  preregistered family.

### ADR-012 addendum 3: Gate v2 result — FAILED on input expansion
(2026-07-14)

Fresh dev seed 8 (FP32 94.0%), 6σ, `scripts/check_stress_gate_v2.py`:

- Early-site reach: **4/4** sites ≥ 1.25× (needed ≥3) — pass.
- Input expansion: **1.96× vs required 2.0× — FAIL by 0.04.**
- Behavioral: NLL degradation **+0.1127** (gate 0.02; 5.6×) — pass.
  Accuracy −3.15 pp, prediction flips 7.05% (reported).
- Pairing — pass.

Mechanistic reading of the miss: the input MinMax ratio is bounded by
(impulse magnitude)/(clean input extreme) = 6σ_img / ~3.06σ_img ≈ 1.96.
The 2× requirement implicitly assumed clean input extremes ≤ 3σ; the
actual calibration extremes are ~3.06σ. The failure is arithmetic
tension between two preregistered numbers (6σ impulses, 2× input
expansion), not a failure of the contamination mechanism, which has now
exceeded its behavioral threshold in three independent runs (seed 7 at
6σ: 4.6×; seed 7 at 10σ: 34×; seed 8 at 6σ: 5.6×).

Disposition: v2 recorded FAILED as written; no threshold adjusted after
data. Per pre-registration the impulse mechanism is closed pending the
user's decision on next steps.

### ADR-012 addendum 4: Impulse Stress Gate v3 preregistration
(2026-07-14, user decision; recorded and committed BEFORE the run)

Standing of prior gates: **Gate v1 and Gate v2 remain FAILED as
originally preregistered.** Neither result is reinterpreted, rescored,
or amended. Gate v3 is a new, prospective mechanism variant — a fresh
experiment, not a reanalysis.

**Sole design change: impulse magnitude 6σ → 7σ.** Every other
generator, model, calibration, quantization, and evaluation setting is
unchanged (fraction 0.002, signs balanced, injected after blur, frozen
0.12 recipe, simulation policy v1, W4A4 behavioral arm).

**The 2.0× input-expansion threshold is retained**, along with all
other v2 criteria verbatim (≥3 of the 4 structural early sites at
≥1.25×; MinMax W4A4 clean-eval NLL degradation > 0.02; pairing
integrity; MinMax-only inspection).

**Rationale**: arithmetic consistency with the measured clean-input
extrema, not optimization against observer performance. Gate v2
measured clean calibration extremes of ~3.06σ, which caps the 6σ input
MinMax ratio at 6/3.06 ≈ 1.96 — the v2 miss. At 7σ the same arithmetic
predicts ≈ 7/3.06 ≈ 2.29 if the fresh split's extremes are similar;
the actual extreme of the untouched seed-6 calibration split has never
been observed, so the gate remains falsifiable (an extreme above 3.5σ
fails it). No robust-observer result informed this choice.

**Fresh evidence base — dev seed 6** (lowest reasonable unused seed):

- Seeds already consumed: 0/1/2 (validation, generator streams 0–5
  including probe streams +3), 7 (Gate v1 + generator screening,
  streams 7–10), 8 (Gate v2, streams 8–11).
- Seeds 3, 4, 5 rejected: each maps at least one of its train/eval/
  calib streams onto validation-seed material (streams 0–5), which
  would let stress design touch validation data.
- Seed 6 uses streams 6 (train), 7 (eval), 8 (calib): no validation
  stream is touched. Streams 7 and 8 previously served as dev-seed
  train/eval material — the same freshness convention already accepted
  when seed 8 followed seed 7 in v2. Critically, no seed-6 stream has
  ever been used as a calibration split or had its MinMax ranges
  inspected, so 7σ cannot have been tuned to this data.
- Stress seed 1006 (1000 + dev seed, per v1/v2 convention).
- A fresh FP32 checkpoint is trained under the frozen benchmark recipe
  into `runs/gen-dev6/` before the gate; its accuracy is recorded.

**Controls**: one Gate v3 attempt only (the runner refuses to execute
if its artifact directory already exists); no fallback magnitude; no
threshold adjustment after seeing results; percentile/MSE-grid/pow2
results stay hidden until the gate passes; validation seeds 0/1/2
remain untouched for the observer-policy comparison.

**Mechanics**: gate criteria are now pure functions in
`quantscope.analysis.stress_gate` (unit-tested, including the exact v2
1.96× failure scenario and threshold boundaries); the frozen v3
constants live there as `GATE_V3_STRESS` / `GATE_V3_SPEC`; the runner
is `scripts/check_stress_gate_v3.py` and writes a provenance-labeled
artifact to `runs/gen-dev6/texture-a-seed6-stress-gate-v3/`.

Pass ⇒ proceed to the approved D observer-policy study on validation
seeds 0/1/2 (ADR-012 design incl. the addendum-2 mechanism
decomposition). Fail ⇒ the impulse stress family is closed permanently
for this phase; stop and report (no automatic switch to glints, no
amended thresholds, no custom-observer runs).

### ADR-012 addendum 5: Gate v3 result — PASSED (2026-07-14)

Fresh dev seed 6 (FP32 95.1% measured), 7σ, single attempt,
`scripts/check_stress_gate_v3.py`, artifact
`runs/gen-dev6/texture-a-seed6-stress-gate-v3/`:

- Early-site reach: **4/4** sites ≥ 1.25× (needed ≥3) — pass
  (input 2.43×, stem_relu 2.18×, block_a.relu1 1.57×,
  block_a.relu_out 1.40×; down_relu 1.23× and all deeper sites ≤1.01×,
  consistent with the v1 attenuation finding).
- Input expansion: **2.43× vs required 2.0× — PASS.** Implied clean
  calibration extreme ≈ 7/2.43 ≈ 2.88σ on the seed-6 split (vs 3.06σ
  on seed 8) — the 7σ arithmetic held with margin.
- Behavioral: NLL degradation **+0.1925** (gate 0.02; 9.6×) — pass.
  Accuracy −5.00 pp, prediction flips 8.20% (reported, not gated).
- Pairing — pass (labels identical; changed-pixel fraction 0.00195).

Verdict computed by the preregistered `GATE_V3_SPEC` criteria with no
adjustment. Cross-gate mechanism record: behavioral damage has now
exceeded its threshold in four independent runs (seed 7 at 6σ: 4.6×;
seed 7 at 10σ: 34×; seed 8 at 6σ: 5.6×; seed 6 at 7σ: 9.6×).

Disposition: **proceed to the approved D observer-policy study** on
validation checkpoints/seeds 0/1/2 under the ADR-012 design (7σ
stress per this addendum, addendum-2 mechanism decomposition included,
no observer-parameter tuning). Robust-observer results were not
inspected before this pass.

### ADR-012 addendum 6: plan step D results — study complete, Q1
CONFIRMED (narrow), Q2 non-inferior, Q3 measured; Q4 deferred
(2026-07-14)

Source of truth: the generated artifacts. Per-seed metrics with
provenance labels in
`runs/validation-012/texture-a-seed{0,1,2}-observer-study/metrics.json`;
cross-seed summary in
`runs/validation-012/observer-study-summary.json`; stress-design gate
evidence in `runs/gen-dev6/texture-a-seed6-stress-gate-v3/`. The
numbers below are transcribed from those artifacts; where a
transcription and an artifact disagree, the artifact wins.

**Gate history** (all preregistered, none reinterpreted): Gate v1
(dev seed 7, 6σ/10σ) FAILED on site coverage — the network-wide
propagation assumption was falsified; behavioral damage passed at both
magnitudes. Gate v2 (dev seed 8, 6σ, structural early-site criteria)
FAILED on input expansion, 1.96× vs 2.0×. Gate v3 (dev seed 6, sole
change 6σ → 7σ, addendum 4) PASSED: input 2.43×, early-site reach 4/4,
behavioral +0.1925 NLL, pairing intact. D ran only after the v3 pass.

**Execution**: frozen validation checkpoints seeds 0/1/2
(freq_step=0.12, `runs/validation-012/`); Gate-v3 impulse stress
(fraction 0.002, 7σ; stress seeds: calibration 1000+seed, evaluation
2000+seed, probe 3000+seed); frozen observers (MinMax; Percentile
0.1/99.9; MSE-grid 32 candidates / 0.3 min fraction; PowerOfTwo
round-up); weights per-channel symmetric MinMax in every arm;
activation calibration observed the weight-quantized model, matching
`simulate_quantized` bit-exactly (unit-tested); no observer-parameter
tuning; single run (`scripts/run_observer_study.py` refuses a rerun).

**Evidence caveats (apply to every number below):**

1. **ReLU-site saturation is not a clipping measure.** Post-ReLU
   activations are mostly exact zeros, which sit on the `qmin` code
   under asymmetric unsigned quantization, so the recorded
   ReLU-site saturation rates (~0.61–0.70 for ALL observers) are
   dominated by ReLU sparsity, not range clipping.
   Saturation-based clipping interpretation is therefore restricted
   to the input site (signed; no zero-pinning artifact). Scale,
   SQNR, and task metrics remain valid at every site.
2. **Every W4A4, W8A4, and W8A8 number in D is SIMULATED** —
   fake-quant simulation policy v1 (FP32 arithmetic with
   fake-quantized weights/activations), not measured integer
   execution, and none of it is hardware/NPU performance.

**Q1 — robustness, primary condition (stressed calibration → clean
evaluation, W4A4): CONFIRMED for both robust observers.**

- Percentile: mean NLL improvement over MinMax **+0.0698**
  (per seed: 0.0610 / 0.0692 / 0.0791); favorable 3/3; mean accuracy
  delta **+3.15 pp** (+2.95 / +3.30 / +3.20). Thresholds: > 0.01
  mean NLL, ≥ 2/3 favorable, accuracy loss ≤ 0.5 pp.
- MSE-grid: mean NLL improvement **+0.0728**
  (0.0601 / 0.0723 / 0.0861); favorable 3/3; mean accuracy delta
  **+3.17 pp** (+2.90 / +3.10 / +3.50).
- Saturation caveat check (input site only, per caveat 1): under
  stressed calibration on the clean probe, percentile/MSE-grid clip
  ≤ 0.10% of clean input values while MinMax clips none — the benefit
  is not driven by catastrophic saturation at any single site. Input
  SQNR (A4) rises from 7.4–8.0 dB (MinMax) to 14.8–16.9 dB
  (percentile) — the mechanism is restored input resolution.
- W4A4 stressed→clean NLL (clean→clean in parentheses), per seed:
  - seed 0: minmax 0.2436 (0.1853); percentile 0.1827 (0.1784);
    mse_grid 0.1836 (0.1757); pow2 0.9185 (0.1930); FP32 clean-eval
    0.1075 measured.
  - seed 1: minmax 0.2493 (0.1782); percentile 0.1801 (0.1719);
    mse_grid 0.1770 (0.1750); pow2 1.6314 (0.2338); FP32 0.1233.
  - seed 2: minmax 0.2314 (0.1525); percentile 0.1523 (0.1405);
    mse_grid 0.1453 (0.1410); pow2 1.6453 (0.1569); FP32 0.1284.

**Preregistered scope limitation — the conclusion is narrow:**

> Percentile and MSE-grid observers improve robustness to controlled
> input-calibration impulse contamination at four-bit activation
> precision. The mechanism decomposition localizes at least 95% of
> the MinMax damage to the input observer, so the result does not
> establish network-wide observer superiority.

Mechanism decomposition (MinMax, W4A4, clean eval; ΔNLL vs clean
calibration): total damage 0.0584 / 0.0711 / 0.0789 per seed;
substituting the stressed **input** qparams alone reproduces 0.0696 /
0.0817 / 0.0751 (≥ 95% of total on every seed; on seeds 0/1 the
input-only substitution slightly exceeds the full-stress damage, i.e.
the remaining sites' stressed qparams partially offset it). No single
ReLU site contributes more than 0.0064. This is
**input/early-activation calibration robustness**, exactly the
narrow reading addendum 2 required.

**Q2 — clean-data non-inferiority (0.005 mean-NLL tolerance, per
configuration, never pooled): non-inferior in all six cells.**
Mean NLL worse-than-MinMax (negative = better):

- Percentile: W4A4 −0.0084; W8A4 −0.0069; W8A8 +0.0005.
- MSE-grid: W4A4 −0.0081; W8A4 −0.0076; W8A8 +0.0005.

At W4A4/W8A4 both robust observers are slightly better than MinMax
even on clean data; at W8A8 the difference is ≤ 0.0008 on every seed.

**Q3 — power-of-two cost (measurement only, round-up mode):**

- clean→clean NLL cost vs MinMax: W4A4 0.0077 / 0.0556 / 0.0044 per
  seed; W8A4 0.0116 / 0.0665 / 0.0009; W8A8 ≤ 0.001 everywhere.
- stressed→clean: W4A4 **+0.67 / +1.38 / +1.41**; W8A4 **+1.28 /
  +2.90 / +1.72**; W8A8 ≤ 0.0013. Round-up doubling of
  already-inflated MinMax ranges is behaviorally catastrophic at
  4-bit activations under contaminated calibration — the largest
  single effect in the study. Power-of-two scale property verified
  exact in every arm.

**Ranking stability (reported, not gated):** stressed→clean W4A4
ranking best→worst — seed 0: percentile, mse_grid, minmax, pow2;
seeds 1 and 2: mse_grid, percentile, minmax, pow2. The robust pair
swaps first/second across seeds (their NLLs differ by < 0.01); the
minmax/pow2 tail is stable. Spearman: 0.8 (s0–s1), 0.8 (s0–s2),
1.0 (s1–s2). Consistent with B's lesson: fine-grained rankings are
seed-sensitive; the robust-vs-baseline separation is not.

**Q4 (optional, non-gating) — DEFERRED to the optional appendix
list.** Rationale: it was explicitly non-gating; C already validated
backend-matched W8A8 arithmetic and lowering; D shows observer-policy
differences at W8A8 are behaviorally negligible (≤ ~0.001 NLL);
running it now would add little to the primary conclusions.

**Plan step D is complete.** Next work package: the reporting phase,
to be separately scoped and approved before work begins.

## ADR-013: Simulated W4A4 quantization-aware fine-tuning (DRAFT
preregistration, 2026-07-15 — committed before any training code;
implementation starts only after user review of this draft)

### Objective and claim boundary

Evaluate whether simulated W4A4 quantization-aware fine-tuning (QAT)
recovers task quality lost by simulated W4A4 PTQ. The primary claim
is strictly numerical: *training with fake quantization can adapt a
checkpoint to the frozen W4A4 quantizer.* No claim of real INT4
execution, INT4 kernels, or latency improvement is made or implied;
every W4A4 number in this ADR's experiments is **simulated**
(fake-quant simulation policy v1). Wall-clock fine-tuning time is
reported as **measured on the development CPU**, with no accelerator
extrapolation.

### Experimental design

Checkpoints: the three frozen freq_step=0.12 FP32 validation
checkpoints (seeds 0/1/2, `runs/validation-012/`). Per checkpoint,
three arms:

1. FP32 baseline (measured; existing artifacts).
2. Simulated W4A4 PTQ baseline: `simulate_quantized` with MinMax
   calibration on the checkpoint's clean calibration split — the
   identical procedure, placements, and qparams as the D study's
   `minmax|W4A4|clean->clean` cell. The QAT run artifact recomputes
   this baseline and must match the D artifact's recorded values
   (consistency check, not a re-derivation).
3. Simulated W4A4 QAT: fine-tuning initialized from the same FP32
   checkpoint, evaluated under the same frozen quantizer (below).

Held identical across arms 2 and 3: quantization placements (policy
v1: input + all ReLU sites, all Conv2d/Linear weights), per-channel
symmetric weights, per-tensor affine activations, the calibration
split and sample order (`texture10_calibration`, seed stream +2,
deterministic), the evaluation split (seed stream +1), preprocessing,
and architecture. QAT is never compared against a PTQ baseline with
different qparams or graph placement.

### Fixed-qparam QAT (this phase's scope)

1. Derive W4A4 qparams with the frozen PTQ calibration procedure
   (MinMax on the clean calibration split, observed against the
   weight-quantized model, exactly as `simulate_quantized`).
2. Insert fake quantization with those qparams and fine-tune the
   weights through a straight-through estimator.
3. **Activation qparams are frozen** for the entire fine-tune and for
   final evaluation; observer ranges are never updated during
   validation training.
4. **Weight fake-quant scale policy (disambiguation, declared now):**
   weight qparams are re-derived per forward pass from the *current*
   weights under the same per-channel symmetric min-max rule, with no
   gradient through the scale computation (scales detached). Frozen
   per-channel weight scales would go stale as weights move; the
   deployment weight quantizer is derived from the final weights at
   export, exactly as PTQ derives it from its input weights. The
   *fixed* quantizer being adapted to is therefore: frozen activation
   qparams + the frozen weight quantization *rule*.
5. Final QAT evaluation: the fine-tuned weights are quantized by the
   same rule and the frozen activation qparams are re-attached — the
   same simulation pipeline that evaluates PTQ.

Observer-updating QAT is future work and MUST NOT be added during
ADR-013 unless separately justified in writing before any validation
result is inspected.

### Implementation boundary

- The NumPy affine core (`quantization/affine.py`) remains the
  backend-independent reference implementation, untouched.
- A new Torch-native differentiable adapter (planned:
  `quantization/qat.py`) implements training-path fake quantization
  with Torch tensor ops only — **no NumPy in the differentiable
  forward path** — consuming QuantScope-generated qparams and
  applying the same clipping/quantization ranges as the simulator.
- Forward parity: the Torch adapter must match the NumPy core's
  quantize/dequantize on deterministic tensors (away from the
  explicitly documented Torch-compatibility differences of ADR-011),
  verified by unit tests before any fine-tuning run.
- **STE gradient policy (declared now): clipped STE.** For
  d(fake_quant(x))/dx: pass-through (1) where the pre-clamp integer
  code lies within [qmin, qmax]; zero where the value saturates.
  Finite gradients through all non-saturated values; both behaviors
  unit-tested at the saturation boundaries.

### Recipe selection (development seed, before validation)

- **Development checkpoint: fresh dev seed 9** — the lowest unused
  experiment seed. Validation seeds 0/1/2 (generator streams 0–5) are
  not used for recipe selection in any way. Seed 9's streams (train
  9, eval 10, calib 11) overlap previously generated dev-seed streams
  (9 = seed 7's calib / seed 6's probe; 10 = seed 7's probe / seed
  8's calib; 11 = seed 8's probe), the same freshness convention
  accepted for gates v2/v3; no validation stream is touched. A new
  FP32 checkpoint is trained under the frozen benchmark recipe into
  `runs/gen-dev9/` before recipe work begins. Reusing dev seed 6 was
  rejected: its checkpoint served Gate v3's stress design, and one
  dev seed per decision is the established pattern.
- **Exactly three predeclared recipes**, varying learning rate only.
  Fixed across recipes: AdamW (the project optimizer), weight decay
  1e-4, batch size 64, cosine schedule, **10 fine-tuning epochs**
  (within the 8–15 bound), the frozen qparam policy above, and
  fake-quant active from the first step (no delayed/gradual
  schedule).
  - R1: lr 3e-4
  - R2: lr 1e-4
  - R3: lr 1e-3
- **Predeclared evaluation order R1 → R2 → R3** (mid magnitude first;
  conservative fallback; aggressive last). Stop at the FIRST recipe
  that, on dev seed 9, improves both W4A4 NLL and W4A4 accuracy over
  that checkpoint's PTQ baseline with no numerical instability (no
  NaN/inf parameter, gradient, loss, or metric). That recipe is
  frozen and run exactly once per validation checkpoint. If none of
  the three passes on the development seed, ADR-013 stops there and
  the negative development result is recorded; no new recipes without
  a written amendment.

### Predeclared metrics

Primary (per checkpoint and mean over seeds 0/1/2):

- ΔNLL: QAT W4A4 vs PTQ W4A4 (simulated).
- Accuracy recovery in percentage points: QAT acc − PTQ acc.
- NLL gap recovery = (PTQ_NLL − QAT_NLL) / (PTQ_NLL − FP32_NLL),
  reported unclipped (values > 1 or < 0 reported as computed).

Secondary: prediction flips vs PTQ and vs FP32; mean correct-class
margin; output SQNR (dB) and cosine similarity vs FP32 logits; weight
and activation saturation diagnostics; per-epoch training loss and
gradient-finiteness checks; elapsed wall-clock fine-tuning time
(measured, development CPU only).

### Success criteria (frozen before any run)

ADR-013 PASSES iff the frozen validation recipe:

1. improves W4A4 NLL over PTQ on ≥ 2 of 3 validation checkpoints;
2. improves mean W4A4 NLL by ≥ 0.01;
3. recovers ≥ 1.0 mean accuracy pp, OR ≥ one-third of the mean
   PTQ-to-FP32 accuracy gap;
4. introduces no NaN or infinite parameter, gradient, loss, or
   metric anywhere in training or evaluation;
5. makes no checkpoint worse than its PTQ baseline by more than
   0.5 accuracy pp.

All checkpoint-level results are reported even if the mean passes.
QAT completion does NOT require reaching FP32 quality. A negative
result is valid and publishable: if the implementation passes parity
and gradient tests but the frozen recipe does not recover PTQ damage,
that is the finding.

### Tests (before any validation run)

Torch/NumPy forward parity on deterministic tensors; STE pass-through
and saturation-zero gradient behavior; clipping/saturation
boundaries; frozen-qparam behavior (activation qparams provably
unchanged across a training step); deterministic fine-tuning on a
tiny fixture (two runs, identical outcome); artifact provenance
labels; actionable failure on incompatible or missing qparams. QAT
training itself is marked slow and stays out of the fast core suite.

### Artifact schema (per QAT run)

`config.json` / `environment.json` (incl. Torch and NumPy versions)
via RunWriter, plus labeled metrics containing: source FP32
checkpoint path and SHA-256; calibration split identity (generator
seed stream and sample count; deterministic order noted); frozen
activation qparams (values + SHA-256 of their serialization) and the
weight-scale rule; the full training recipe; per-epoch loss/metric
series; final FP32/PTQ/QAT comparison with the predeclared metrics;
measured/simulated labels on every entry; elapsed CPU seconds
(measured).

### Scope exclusions

No real-INT4 execution claims or kernels; no Q4; no W3A3; the Torch
2.2.2 parity guard stays; no hardware cost model work; no Texture-10
generator changes; no observer-policy retuning; B/C/D conclusions
stay as recorded.

### Status

DRAFT — awaiting user review. Implementation (Torch adapter, tests,
training loop, dev-seed-9 checkpoint) begins only after approval;
validation seeds are touched only after the dev recipe freezes.

### ADR-013 addendum: approved with amendments (2026-07-15) —
preregistration final; implementation authorized

The draft is approved. Decisions 1 (dev seed 9), 3 (clipped STE,
applied and tested identically for weights and activations), and 4
(three LRs, first-pass selection, everything else frozen) stand as
written. Two amendments and two freezes:

1. **Terminology**: this phase is **fixed-quantization-specification
   QAT**, not fully fixed-qparam QAT. Only activation qparams are
   numerically frozen; the weight scheme/range/rule is frozen while
   per-channel weight scales are recomputed from current weights
   (detached from autograd). After fine-tuning, final weight qparams
   are recomputed once for the exported/evaluated QAT checkpoint and
   stored in the artifact.
2. **Baseline consistency (amended decision 5)**: the D artifact is
   the canonical PTQ baseline; the QAT run independently recomputes
   it as a consistency gate with predeclared tolerances — accuracy
   and prediction counts exactly equal; NLL within absolute 1e-6;
   other aggregate float metrics within the existing
   deterministic-test tolerances; checkpoint identity, calibration
   sample IDs, qparam-policy identifier, observer configuration,
   quantization ranges, and evaluation sample IDs exactly equal.
   Exceeding any tolerance STOPS the study before QAT training as a
   provenance/reproducibility failure; canonical D values are never
   overwritten. Both canonical and recomputed values are stored in
   the QAT artifact.
3. **Checkpoint-selection freeze**: the epoch-10 checkpoint is used
   for development screening and every validation run. No
   retrospective best-epoch selection, no early stopping, no
   metric-based restore. Per-epoch metrics are diagnostic only. The
   selected learning rate + fixed 10 epochs is the frozen validation
   recipe.
4. **Fake-quant scheduling freeze**: weight and activation fake
   quantization are enabled from the first fine-tuning step through
   the last. No FP32 warm-up, no delayed activation quantization, no
   observer updates, no staged bit widths, no batch-dependent
   recalibration — those are separate QAT variants and would weaken
   the direct PTQ-vs-QAT comparison.

Authorized implementation order: (1) Torch-native affine fake-quant +
STE tests; (2) NumPy/Torch forward-parity tests; (3) tiny
deterministic QAT fixture; (4) seed-9 FP32 checkpoint; (5) the
preregistered development recipe sequence; (6) validation on seeds
0/1/2 only if a development recipe passes. The hardware cost model,
regression harness, W3A3, Q4, and newer-Torch validation stay out of
scope for ADR-013.

### ADR-013 addendum 2: results — PASSED on all five criteria
(2026-07-15)

Development phase (seed 9): R1 (lr 3e-4) passed on the first attempt
(ΔNLL −0.0610 vs PTQ, +2.20 pp, gap recovery 0.815) and was frozen
per the first-pass rule; R2/R3 were never run. Artifact:
`runs/gen-dev9/texture-a-seed9-qat-dev-lr0.0003_ep10/`.

Validation: baseline consistency gates PASSED on all three seeds
(accuracy and prediction counts exact, NLL within 1e-6, per-site
activation scales exactly equal to the canonical D artifact values;
both canonical and recomputed values stored). The frozen recipe
(lr 3e-4, 10 epochs, epoch-10 checkpoint, fake-quant first step
through last) then ran exactly once per checkpoint:

| seed | FP32 NLL / acc (measured) | PTQ W4A4 NLL / acc | QAT W4A4 NLL / acc | ΔNLL | acc rec | gap recovery |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.1075 / 95.70% | 0.1853 / 92.55% | 0.1122 / 95.25% | −0.0731 | +2.70 pp | 0.940 |
| 1 | 0.1233 / 95.40% | 0.1782 / 93.45% | 0.1328 / 95.00% | −0.0454 | +1.55 pp | 0.828 |
| 2 | 0.1284 / 95.00% | 0.1525 / 94.00% | 0.1400 / 94.55% | −0.0125 | +0.55 pp | 0.518 |

All W4A4 values simulated (fake-quant policy v1, not integer
execution); FP32 measured.

Criteria: (1) NLL improved on 3/3 (needed ≥2/3); (2) mean NLL
improvement 0.0437 (needed ≥0.01); (3) mean accuracy recovery
1.60 pp (needed ≥1.0 pp or ≥⅓ of the 2.03 pp mean PTQ gap — both
met); (4) no non-finite parameter/gradient/loss/metric (enforced by
per-step raises during training); (5) no checkpoint worse than PTQ
(all three improved). **ADR-013 PASSES.**

Checkpoint-level honesty: the effect size tracks the size of the PTQ
gap — seed 2, with the smallest PTQ damage (1.00 pp), recovered least
(+0.55 pp, gap recovery 0.518). Mean gap recovery 0.762; no
checkpoint reached FP32 quality, which ADR-013 did not require.
Secondary evidence: QAT roughly halves prediction flips vs FP32
relative to PTQ's flips and raises logit SQNR vs FP32 on every seed
(13.5→19.2 / 15.8→20.2 / 18.9→20.8 dB, simulated). Fine-tuning
wall-clock ≈151 s per checkpoint, measured on the development CPU —
no accelerator claim.

Supported claim (exactly as preregistered): *training with fake
quantization can adapt a checkpoint to the frozen W4A4 quantizer* —
simulated throughout; no real INT4 execution or latency claim.
Artifacts: `runs/validation-012/texture-a-seed{0,1,2}-qat-w4a4/`,
`runs/validation-012/qat-study-summary.json`.

## ADR-014: analytical hardware cost model + mixed-precision
recommendations (DRAFT preregistration, 2026-07-15 — committed before
any cost-model code; implementation starts only after user review)

### Objective and claim boundary

Replace the weight-bits proxy (`search.exhaustive.config_cost`) with a
validated, profile-driven analytical cost model, and use it to
generate mixed-precision recommendations from the existing B3
exhaustive sweep tables. Every hardware cost and recommendation is
**estimated**; no latency, energy, or throughput value may be
described as measured. The intended claim is exactly:

> QuantScope can consume an explicit hardware profile, calculate
> transparent component-wise estimated costs, and show how hardware
> assumptions change mixed-precision recommendations.

The claim is NOT that `generic_edge_npu` is accurate or represents
any commercial accelerator. The profile remains fictional. Agreement
with the old proxy is a valid, documentable result.

### Frozen experimental inputs (no reruns)

The three B3 sweep tables
(`runs/validation-012/texture-a-seed{0,1,2}-sweep/sweep_table.json`):
256 assignments per checkpoint over the frozen eight policy-v1
quantization groups, each group W4A4 or W8A8 (the sweep assigned
`SimQuantConfig(b, b)` per group), with simulated NLL/accuracy. No
training, inference, calibration, or quantization is rerun to build
the cost model.

### Hardware profile schema (Pydantic, versioned)

`configs/hardware/generic_edge_npu.yaml` is the initial profile but
its CURRENT contents are not assumed valid; it will be rewritten to
the schema below (existing fields preserved only where they fit).

Proposed schema (`HardwareProfile`, `schema_version: 1`):

- `name: str`, `schema_version: int` (only 1 accepted),
  `description: str`, `fictional: bool` (must be true for this
  profile; a disclaimer string is required),
  `assumptions: list[str]` (non-empty).
- `unit: str` — the normalized cost unit. Declared definition:
  **1 ncu = the estimated cost of one INT8×INT8 MAC on this
  profile.** Every coefficient carries this unit via its field name.
- `compute_ncu_per_mac: dict[str, float]` keyed by `"W{w}A{a}"` —
  a coefficient for each supported (weight_bits, activation_bits)
  pair. Proposed fictional values: W8A8 1.0, W4A4 0.55, W4A8 0.75,
  W8A4 0.80 (the sweep uses only W8A8/W4A4; the extra pairs
  exercise the schema).
- `weight_memory_ncu_per_bit: float` (proposed fictional 0.02).
- `activation_memory_ncu_per_bit: float` (proposed fictional 0.02).
- `per_layer_overhead_ncu: float` (proposed 0.0; the field is
  explicit so overhead is only ever counted when the profile
  declares it).
- `accumulator_bits: int` (32; recorded, NOT used by the v1
  calculation — documented).
- `supported_weight_bits: list[int]`, `supported_activation_bits:
  list[int]` (both [4, 8]).

Validation rejects: unknown precision pairs referenced by a model
configuration; pairs in the table not covered by supported bit lists;
negative, NaN, or infinite coefficients; missing units (enforced by
field naming + `unit`); duplicate pair keys; `schema_version != 1`;
profiles missing any pair a scored configuration needs; and
`fictional: false` for this profile file.

### Model accounting (deterministic analysis pass)

One shape-instrumented forward pass of the FP32 BottleneckResNet at
the benchmark input size (1×1×32×32) records, per quantization group:
parameter count; MAC count (conv: out_elems × in_C × kH × kW; linear:
in × out); input/output activation element counts and tensor shapes;
the assigned (weight_bits, activation_bits); and the explicit
exclusion list. Repeated runs must produce identical accounting; the
accounting JSON is hashed (SHA-256) into every downstream artifact.

**Declared traffic assumption (v1, visible in artifacts and docs):**
no-cache, single-read/single-write per distinct tensor. Each
quantized activation tensor is counted ONCE for write and ONCE for
read at the precision of its *producing* group, with no
per-consumer multiplier — so activations shared by residual paths
are not double-counted. The model input is counted as one read at
the stem group's activation precision (the stem group owns input
quantization). Consumer groups add no read terms of their own.

**Documented exclusions (recorded in every artifact):** BatchNorm
(unfolded under policy v1), residual adds, pooling, flatten, and the
classifier logits (unquantized float island under policy v1) carry
no compute coefficient and no activation-memory term; they are
constant across all 256 configurations and excluded from totals.

### Cost equations (component-wise; never one opaque score)

For group g with assignment (w_g, a_g):

- `compute_g = MACs_g × compute_ncu_per_mac["W{w_g}A{a_g}"]`
- `wmem_g = weight_elements_g × w_g × weight_memory_ncu_per_bit`
- `amem_g = Σ_{tensors produced by g} 2 × elements × a_g ×
  activation_memory_ncu_per_bit` (+ the input-read term for the stem
  group: `elements × a_stem × activation_memory_ncu_per_bit`)
- `overhead_g = quantized_layer_count_g × per_layer_overhead_ncu`
- `total_g = compute_g + wmem_g + amem_g + overhead_g`
- `model_total = Σ_g total_g` — a plain declared sum; all four
  components are reported per group and per model in every artifact.

Normalization: costs are reported both in raw ncu and normalized to
the all-INT8 configuration of the same checkpoint (all-INT8 ≡ 1.0
exactly). The all-INT4 normalized cost is also reported, and every
mixed configuration must lie in [all-INT4, all-INT8] normalized cost
(no documented float-island exception applies to totals, since the
excluded constants are outside the sum).

### Predeclared invariant tests (internal consistency, not realism)

1. All-INT4 total < all-INT8 total under `generic_edge_npu`.
2. Lowering one group from (8,8) to (4,4) with all others held fixed
   strictly reduces that group's compute, weight-memory, and
   activation-memory components, and never increases any other
   group's components.
3. `model_total` equals the sum of the reported components (exact,
   same float path).
4. Repeated analysis is deterministic (identical accounting JSON and
   hashes across runs).
5. Per-group totals reconcile with the whole-model total.
6. Unsupported precision assignments fail loudly (actionable error).
7. Schema validation rejects each malformed-profile case listed
   above (one test per rejection rule).

### Mixed-precision recommendations

Per checkpoint (never averaged into a universal assignment):

1. Recompute the exact NLL-vs-estimated-cost Pareto frontier over all
   256 configurations under the new cost model.
2. For each prospective normalized-cost budget **0.60, 0.75, 0.90**:
   recommend the configuration with the lowest simulated NLL whose
   normalized estimated cost is ≤ the budget. If no configuration is
   feasible, report infeasibility explicitly (never silently pick the
   cheapest).
3. Exact-NLL ties break deterministically: (a) lower estimated cost;
   (b) higher simulated accuracy; (c) lexicographically smallest
   configuration identifier, defined as the per-group bits tuple
   joined with "-" in frozen group-dict order (e.g.
   "4-8-8-4-4-8-8-4").

### Comparison with the old weight-bits proxy (per checkpoint)

- Spearman rank correlation across all 256 configurations
  (new normalized cost vs old proxy cost).
- Pareto-frontier membership changes (set difference + Jaccard).
- Recommendation changes at each budget (old-cost budgets evaluated
  at the same 0.60/0.75/0.90 levels for the comparison).
- Concrete examples where activation-memory or compute terms reorder
  configuration pairs that the weight-bits proxy tied or ordered
  oppositely.

### Success criteria (frozen)

ADR-014 passes iff: the schema and the rewritten generic profile
validate; model accounting reconciles deterministically; all invariant
tests pass; all 256 configurations per checkpoint receive transparent
component-wise costs; checkpoint-specific recommendations exist for
every feasible budget (with explicit infeasibility reports otherwise);
at least one artifact records the traffic assumption, exclusion list,
profile SHA-256, accounting SHA-256, and provenance labels; every cost
and recommendation is labeled **estimated**; and no measured
performance claim is introduced anywhere. The new model is NOT
required to disagree with the old proxy.

### Deliverables and artifact format

- `src/quantscope/hardware/`: `profile.py` (Pydantic schema +
  loader/validator), `accounting.py` (model analysis pass),
  `cost.py` (component calculation + normalization).
- CLI: `quantscope hw-validate --profile <yaml>` (validate + print a
  summary) and `quantscope hw-score --seed N --bits 4,8,...`
  (component-wise cost for one assignment).
- `scripts/run_hwcost_study.py`: enriches the existing B3 sweep
  artifacts WITHOUT modifying the originals — writes
  `runs/validation-012/texture-a-seed{s}-hwcost/` via RunWriter
  (kind="hwcost") containing `hwcost_table.json` (per-config
  component costs, raw + normalized), the recomputed Pareto frontier,
  the budget recommendations, the proxy comparison, the assumptions
  block, and the profile/accounting hashes; every metric labeled
  ESTIMATED (task metrics quoted from B3 stay labeled SIMULATED).
- Unit + integration tests per the invariant list; fast and offline.
- ADR-014 results addendum; report/README alignment AFTER results.

### Scope exclusions

No accelerator latency benchmarking; no fitting coefficients to task
results; no claim that generic_edge_npu represents a commercial
accelerator; no B3 inference reruns; no numerical-regression harness;
no Q4 or W3A3; the Torch 2.2.2 guard stays; QAT, observer, and
generator conclusions unchanged.

### Status

DRAFT — awaiting user review. The hardware module is not implemented
until this preregistration is approved.

### ADR-014 addendum: approved with amendments (2026-07-15) —
preregistration final; implementation authorized

The draft is approved; the fictional coefficients stand as
assumptions, not calibrated hardware facts. Amendments:

1. **Profile migration**: `configs/hardware/generic_edge_npu.yaml` is
   rewritten to schema v1 (single canonical profile, no ambiguous
   authority). The old throughput/bandwidth-format contents are
   preserved verbatim as
   `tests/fixtures/hardware/generic_edge_npu_legacy_v0.yaml`, and a
   test proves the legacy format fails validation with a clear
   missing/unsupported-schema error.
2. **Schema controls**: Pydantic with unknown fields forbidden;
   explicit `schema_version`, `profile_name`, `fictional: true`,
   description/assumptions, coefficient-unit definitions, supported
   precision pairs, accumulator precision, and a traffic-model
   identifier. Precision coefficients are a LIST of entries (not a
   YAML mapping, where duplicate keys are silently overwritten);
   (weight_bits, activation_bits) uniqueness is validated.
3. **Numerical behavior**: all cost math in float64; no rounding
   before totals, normalization, Pareto construction, feasibility, or
   tie-breaking (rounding is display-only); serialized precision
   sufficient to reproduce ordering; both the raw source-file SHA-256
   and a canonical parsed-profile digest are recorded.
4. **Canonical group ordering**: an explicit versioned constant
   (`GROUP_ORDER_V1`, matching the frozen B3 partition) replaces
   incidental dict order; it is stored in every accounting and
   recommendation artifact; configuration identifiers are readable
   (`stem=w4a4|block_a_conv1=w8a8|...`) and used for the final
   lexicographic tie-break; a B3 artifact whose group set does not
   exactly match fails loudly.
5. **Activation accounting clarified** (base assumption approved):
   no cache; one read and one write per distinct modeled activation
   tensor; no per-consumer multiplication; shared residual tensors
   counted once; the model input is owned by the stem group (one
   read); host transfer and input-quantization overhead excluded.
   Reconciliation tests use tensor/graph-node identities, not only
   aggregate element counts.
6. **Classifier and exclusions**: classifier weights and MACs are
   modeled at the group's assigned precision and its input traffic is
   modeled (at the producing expand group); only the unquantized
   output logits are excluded from activation-memory cost. Excluded
   operations (BatchNorm, residual adds, pooling, flatten, logits,
   quantize/dequantize boundaries) are recorded explicitly with
   counts. Normalized cost is described as the **normalized estimated
   cost of the modeled quantizable workload** — never whole-model
   energy/latency/total hardware cost. Every artifact carries modeled
   weighted-module count, modeled MAC/parameter totals, excluded op
   types and counts, traffic assumptions, and a warning that omitted
   constant costs can overstate the fraction of system-level savings.
7. **Normalization/budget semantics**: normalized_cost =
   configuration_total / all_int8_total (same profile + accounting);
   all-INT8 asserted 1.0 within tolerance; budgets apply to this
   normalized modeled-workload cost. Infeasible budgets emit a
   structured result (feasible: false; cheapest normalized cost;
   cheapest configuration identifier; no recommendation); budgets are
   never relaxed.
8. **Additional invariants**: components finite and nonnegative;
   zero-MAC/zero-parameter groups behave; changing one group's
   precision changes only that group's precision-dependent
   contributions; group totals sum to configuration totals within
   tolerance; profile/accounting hashes stable across runs; different
   coefficients change digest and costs; unsupported W/A combinations
   fail before any partial artifact is written; monotonicity tested
   at both group-component and whole-configuration level.
9. **Artifact lineage**: the enrichment script never modifies B3
   files; each artifact records source B3 path + SHA-256, checkpoint
   identity, profile path/source hash/canonical digest, accounting
   digest, group-order version, cost-model schema version, all
   component costs, raw + normalized totals, and provenance (quality
   simulated; costs estimated; recommendations derived from simulated
   quality and estimated cost). Data files are written to a temporary
   file and atomically renamed only after complete validation.
10. **Proxy interpretation**: the weight-bits proxy remains a
    historical baseline; B3 artifacts are never rewritten or
    retrospectively presented as hardware-profile costs. Agreement
    and low correlation are both valid outcomes (the new model adds
    compute and activation terms the proxy omitted).

Authorized order: profile schema/loader → legacy rejection test →
deterministic accounting → component-wise cost → CLI validation and
scoring → B3 enrichment/recommendation script → invariant/
integration/provenance/determinism tests → results addendum →
README/report alignment. The numerical-regression harness waits until
ADR-014 closes.

### ADR-014 addendum 2: results — PASSED all success criteria
(2026-07-16)

Artifacts: `runs/validation-012/texture-a-seed{0,1,2}-hwcost/`
(per-config component tables written atomically with full lineage) and
`runs/validation-012/hwcost-study-summary.json`. Profile canonical
digest c4bfa0c4…, accounting digest 45d0471b…, group-order-v1,
traffic model single-read-single-write-per-tensor-v1. B3 originals
untouched.

- Schema + rewritten generic profile validate; the legacy v0 format is
  preserved as a fixture and provably rejected.
- Accounting reconciles deterministically (tensor-identity tests) and
  all invariant tests pass (31 hardware tests).
- All 256 configurations per checkpoint carry component-wise estimated
  costs; all-INT8 normalized cost is exactly 1.0; all-INT4 is 0.5498;
  every mixed configuration lies within bounds.
- **All nine checkpoint×budget cells feasible** (all-INT4 0.5498 <
  0.60): e.g. seed 0 recommendations — budget 0.60: NLL 0.1404 / acc
  94.45% at cost 0.5599; 0.75: NLL 0.1144 / 95.50% at 0.7497; 0.90:
  NLL 0.1115 / 95.75% at 0.8728 (NLL/accuracy simulated, costs
  estimated). Recommendations are checkpoint-specific and never
  averaged.
- **Proxy comparison** (historical weight-bits baseline, unrewritten):
  Spearman ρ = 0.8858 — identical on all three checkpoints because
  costs depend only on the assignment, not the checkpoint. Pareto
  membership shifts: Jaccard 0.609 / 0.462 / 0.300 (seeds 0/1/2);
  recommendations changed in 7 of 9 checkpoint×budget cells. Concrete
  reversal: two configurations tied at old proxy cost 0.9296 separate
  to 0.8667 vs 0.8729 normalized — quantizing the stem (few weights,
  large early activation maps) is nearly free under the weight-bits
  proxy but saves real activation traffic and compute under the
  profile. High-but-not-perfect correlation is exactly the expected
  behavior: the new model adds compute and activation terms the proxy
  omitted.
- No measured performance claim anywhere; every cost labeled
  estimated; the supported claim is exactly the preregistered one
  (QuantScope consumes an explicit profile, computes transparent
  component-wise estimated costs, and shows how hardware assumptions
  change recommendations — under a fictional profile whose
  coefficients are assumptions).

**ADR-014 PASSES.** The weight-bits proxy remains in the historical B3
artifacts as recorded.

## ADR-015: numerical-regression harness (DRAFT preregistration,
2026-07-16 — committed before implementation; NOT pushed; awaiting
review)

### Objective

A deterministic, provenance-aware numerical-regression harness that
compares a newly generated artifact against a versioned, committed
baseline specification. It must detect numerical regressions, detect
structural regressions, distinguish regressions from malformed or
incompatible inputs, produce a readable and machine-readable diff,
and never update a baseline implicitly. Baseline updates are code-
review decisions.

### Initial CI smoke artifact

One small, deterministic, offline artifact generated by a package
module (`quantscope.regression.smoke`) with NO training and NO
dependence on historical `runs/` directories (B/C/D/QAT/hwcost
artifacts remain historical evidence, never CI fixtures):

- Fixed seeds throughout (`torch.manual_seed(0)`; synthetic dataset
  seed 0); TinyCNN (4 classes, 16×16) freshly initialized —
  regression targets determinism, not task quality.
- Contents: deterministic FP32 evaluation metrics on 64 fixed samples
  (correct-count, NLL, mean margin — **measured**); simulated W8A8
  AND W4A4 metrics via `simulate_quantized` with 32 fixed calibration
  samples (**simulated**); representative per-site activation scales
  and zero points at both widths; per-site saturation diagnostics on
  the calibration batch; one estimated hardware-cost calculation
  (benchmark BottleneckResNet accounting + all-INT4 normalized cost
  under the canonical schema-v1 profile — **estimated**; needs no
  trained weights). Estimated runtime well under 10 s; if CI shows
  the pair of quantization configs threatens the sub-minute core
  budget, the preregistered fallback is dropping W8A8 (keeping FP32,
  W4A4, qparams, provenance labels, and the cost result).
- Artifact format: one JSON document with `artifact_type:
  "regression-smoke"`, `artifact_schema_version: 1`, an `environment`
  block, and a `sections` tree whose leaves are
  `{value, provenance}` pairs. Written atomically.

### Baseline location and schema

Committed baselines live under `tests/baselines/`. Pydantic schema,
unknown fields forbidden, `baseline_schema_version: 1`:

- `baseline_name`, `description`, `artifact_type`,
  `compatible_artifact_schema_versions: list[int]`,
  `capture_command` (string, reproducibility note),
  `environment_rules` (below), `rules: list[CheckRule]`,
  `canonical_digest` (SHA-256 of the key-sorted baseline body
  excluding the digest field itself; validated on load),
  `quantscope_commit` (report-only traceability — never a
  compatibility gate).
- `CheckRule`: `path` (JSON Pointer), `comparator`
  (`exact` | `close` | `no_worse` | `structure`), comparator-specific
  fields (below), `expected`, `provenance` (required for every
  checked quantity), `rationale` (required whenever any tolerance is
  nonzero).
- `ignored_paths: list[str]` — explicitly ignored volatile paths.
  Timestamps, elapsed wall-clock, absolute/temporary paths, machine
  names, process IDs, random run-directory names, and git commit
  hashes are never numerically checked.
- Compatibility gating uses artifact-schema version, quantization-
  policy version, hardware-cost schema version, observer/qparam-
  policy identifiers, and model/configuration identifiers — not
  commit hashes.

### Field-path syntax

**JSON Pointer (RFC 6901)**: `/sections/w4a4/nll/value`. Mappings by
key; sequences by decimal index; `~0`/`~1` escaping for `~` and `/`.
A path absent from the artifact is a **regression (exit 1)** when it
is a checked path in a compatible artifact, and a **harness error
(exit 2)** when the absence indicates an incompatible artifact type
or schema (classified before rule evaluation). Duplicate rules for
one path are rejected at baseline-validation time (exit 2); there is
no precedence mechanism in v1. Rules evaluate in baseline order;
reports sort by path for determinism.

### Comparators

1. **exact** — strings, integers, booleans, identifiers, sample and
   prediction counts, provenance labels, configuration names, zero
   points.
2. **close** — `abs(actual − expected) <= atol + rtol *
   abs(expected)`, computed in float64; the diff records expected,
   actual, absolute and relative differences, atol, and rtol. NaN and
   infinity are rejected (exit 1 as a value regression when the
   baseline expected a finite value; the initial baseline permits no
   non-finite values anywhere).
3. **no_worse** — declares `direction: higher_is_better |
   lower_is_better` and `degradation_atol >= 0`; passes when the
   metric is no worse than `expected` by more than `degradation_atol`
   (boundary inclusive: exactly at tolerance passes); improvements
   always pass and are reported.
4. **structure** — for mappings: `required_keys`,
   `allow_extra_keys: bool` (default false); for sequences:
   `required_length` and/or ordered element rules,
   `allow_extra_elements: bool` (default false). Extra fields inside
   checked sections are rejected by default.

### Regression vs harness-error classification (exit codes)

- **0** — pass.
- **1** — regression: numerical tolerance failure; no_worse breach;
  structural key/length mismatch; missing required field in an
  otherwise compatible artifact; unexpected extra checked field;
  provenance-label change (fails even when the value is unchanged).
- **2** — harness/input error: invalid JSON/YAML; unsupported
  baseline schema; unsupported artifact schema; wrong artifact type;
  unreadable or nonexistent input; invalid comparator definition;
  ambiguous/duplicate path rules; digest mismatch in the baseline.

### Provenance enforcement

Every checked quantity carries an exact provenance expectation using
the project's meanings: FP32 task metrics **measured**; fake-quant
metrics **simulated**; analytical hardware costs **estimated**;
recommendations derived from simulated quality + estimated cost.

### Environment metadata

Recorded in every artifact and diff: Python, Torch, NumPy, QuantScope
versions, platform identifier. Three tiers in `environment_rules`:

1. **exact gates**: torch == 2.2.2 (the ADR-011 guard's validated
   environment; the guard itself stays);
2. **allowed sets**: Python minor in {3.11, 3.12} (CI runs both;
   never an exact-match gate); numpy in {1.26.*};
3. **report-only**: platform/machine identity (promoted to a gate
   only if the smoke artifact proves platform-sensitive, as a
   documented amendment).

### CLI

Typer sub-app (`quantscope regression …`):

- `quantscope regression validate-baseline BASELINE` — schema +
  digest + rule-uniqueness validation.
- `quantscope regression check ARTIFACT --baseline BASELINE
  [--diff-out FILE]` — never writes baselines; terminal output gives
  the verdict, failure counts by category, the most important failed
  paths, and the diff-artifact location.
- `quantscope regression capture ARTIFACT --output BASELINE
  [--overwrite]` — refuses to overwrite without the explicit flag;
  records the candidate artifact digest; validates the proposed
  baseline before writing; writes atomically; on overwrite emits a
  reviewable comparison against the previous baseline; never runs in
  CI.
- `quantscope regression smoke --out ARTIFACT` — the deterministic
  smoke generator.

### Deterministic diff artifact

JSON containing: `diff_schema_version: 1`; baseline identity +
canonical digest; candidate artifact digest; overall verdict; failure
category; environment metadata; and one entry per checked path with
comparator type, expected, actual, tolerances, absolute/relative
differences where applicable, pass/fail, and a concise explanation.
Determinism: canonical (sorted) field ordering; checked paths sorted;
no timestamps in the canonical diff; no absolute paths; normalized
path separators; floats serialized via `repr` round-trip (identical
on Python 3.11/3.12 — both use shortest-repr); atomic
temp-file-then-rename writes. Candidate artifacts need not be
byte-identical across Python versions — checked values must be
semantically identical within the preregistered comparator rules.

### Initial tolerance policy (per family; every nonzero tolerance
carries this rationale in the baseline)

| Family | Rule | Rationale |
| --- | --- | --- |
| identifiers, labels, provenance, config names | exact | categorical |
| sample/prediction counts, zero points | exact | integers |
| accuracy | exact correct-count (integer) | fixed finite sample set; the fraction is derived |
| NumPy-derived activation scales | close, atol 1e-9, rtol 1e-7 | stored float32, deterministic given pinned numpy; headroom only for libm-level platform variation, far below any real regression |
| Torch-derived float metrics (NLL, mean margin) | close, atol 1e-6, rtol 1e-5 | float32 reductions may reorder across CPU/BLAS builds within a few ulps |
| SQNR (dB) | close, atol 1e-4 | log of float32-accumulated ratio |
| cosine similarity | close, atol 1e-7 | bounded near 1; float64 final math |
| saturation rates | close, atol = 0.5 / element_count | quotients of integer counts over fixed sizes; any count change fails |
| normalized hardware cost | close, atol 1e-12, rtol 0 | pure float64 arithmetic on integer counts × declared coefficients |

No global tolerance exists. Tolerances are never loosened merely to
make 3.11 and 3.12 agree — any cross-version difference is first
quantified and recorded.

### CI integration

One added step per Python version, after the fast suite, inside the
pinned torch 2.2.2 environment: generate the smoke artifact →
`regression validate-baseline` → `regression check` → print/upload
the structured diff on failure. Offline, no training, comfortably
inside the core-suite time budget, no historical run directories.

### Preregistered tests

Valid baseline parsing; unknown-field rejection; duplicate/ambiguous
rule rejection (exit 2); exact pass/fail; atol boundary; rtol
boundary; combined atol+rtol behavior; no_worse in both directions
incl. the boundary; required-field failure (exit 1);
unexpected-extra-field failure (exit 1); provenance mismatch with
unchanged value (exit 1); NaN/inf rejection; malformed baseline
(exit 2); incompatible artifact type/schema (exit 2); deterministic
path ordering; byte-deterministic diff output; atomic writes; exit
codes 0/1/2 each demonstrated end-to-end; capture overwrite
protection (+ explicit flag + reviewable comparison); complete
smoke-generate-then-check round trip; 3.11/3.12 compatibility (via
CI); and a **deliberate perturbation test**: one realistic numerical
field (a W4A4 NLL) perturbed beyond tolerance must fail with a diff
identifying the exact path, expected and actual values, tolerances,
and the `regression` category.

### Success criteria (frozen)

Schema validates; the smoke artifact is deterministic and offline;
the committed baseline passes on 3.11 and 3.12 CI; a deliberately
perturbed artifact fails with a useful structured diff; numerical,
structural, provenance, and incompatible-input failures are
distinguished by exit code and category; baseline updates cannot
happen implicitly; the fast suite stays within the sub-minute target;
CI remains green; README and PROGRESS no longer list the harness as
missing; a clean-clone verification succeeds.

### Deliverables

`src/quantscope/regression/` — `models.py` (schemas), `compare.py`
(comparators + classification), `diff.py` (deterministic diff),
`capture.py` (capture workflow), `smoke.py` (artifact generator);
CLI sub-app in `cli.py`; `tests/baselines/smoke.json`;
`tests/unit/regression/` + an integration round-trip test; CI
workflow step; ADR-015 results addendum; README/PROGRESS closeout.

### Scope exclusions

No B/C/D/QAT/hwcost reruns; no performance benchmarking; no W3A3; no
Q4; the Torch 2.2.2 guard stays; previous findings unchanged; no
harness code before this draft is approved; the draft is not pushed
before approval.

### Status

DRAFT — awaiting user review before any regression-harness code.

### ADR-015 addendum: approved as drafted (2026-07-16) —
implementation authorized

The draft is approved without amendment: JSON Pointer paths, the
two-config smoke artifact (W8A8 + W4A4, with the preregistered
fallback), the untrained seeded TinyCNN target, the reused
BottleneckResNet accounting for the cost component, and the tolerance
table all stand. One implementation clarification recorded now:
`provenance` is required on every metric rule (exact/close/no_worse
against `{value, provenance}` leaves); plain identity strings (e.g.
`/identifiers/*`, digests) may use exact rules with no provenance
field, since they are configuration facts rather than measured/
simulated/estimated quantities. Environment-gate violations classify
as harness/configuration errors (exit 2): running under the wrong
torch is an incompatible input, not a code regression.
