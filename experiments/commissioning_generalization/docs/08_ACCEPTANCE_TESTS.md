# 08. Acceptance tests and release checklist

These are software/scientific-integrity tests, not tests that the proposed method wins.
Each must have actual executable coverage, saved output, and an explicit pass/fail/block.
Use tolerance1e-6 for FP32 inference comparisons (relax documented BF16 comparisons to
1e-3); budget/ID/byte/count invariants are exact. Disable dropout for invariance tests.

## Repository and protocol

A01: all protected legacy file hashes unchanged before/after implementation.
A02: legacy five unit tests execute in isolated original directory and pass, or clearly
     document an existing failure rather than editing old code.
A03: new code/imports/output paths stay in the new directory/external data root.
A04: config counts/unique seeds/family priors match; schema rejects unknown hidden fields.
A05: no formal run begins without dependency, asset, manifest and protocol hashes.
A06: failed/unfinished jobs cannot create a success marker; resume checks hashes.

## Executed system

S01: healthy canonical chain and split/merge fixtures reach task requirements.
S02: with fixed exogenous seeds, cache fault changes actual controller input and executed
     commands; it is not just a label/log modification.
S03: queue fault changes actual actuator delivery times in simulation time; host sleeps
     do not implement the effect.
S04: adapter fault changes the actual payload using the injected gain; nominal metadata
     stays unchanged and does not reveal the injected parameter.
S05: controller computation demonstrably depends on delivered measured state. Injecting
     stale input must not be irrelevant because u simply equals q_ref.
S06: ordinary repair of correct node restores its configuration; repair of a wrong node
     does not magically clear the true fault. Fault severity may remain hidden.
S07: every graph's routing preserves required joints and representation contracts.
S08: structural nodes affect execution, not decorative GNN-only nodes; clock nodes are
     explicitly representational and not counted as executed components.
S09: same source/task seed and disabled faults reproduce nominal physics across probe
     choices. Logging changes wall time but not simulator dynamics in this v1.

## Clocks, probes and cost

O01: recorded timestamps equal private event time+domain offset+independent error; only
     the observable projection crosses the API boundary.
O02: paired records can match by opaque IDs only when both were acquired. No missing
     endpoint can be reconstructed from evaluator full logs.
O03: four-timestamp calibration estimator matches its algebra in zero-delay-asymmetry
     fixture; main asymmetric/noisy links yield nonzero residual variance.
O04: calibration does not alter true clocks or repair/control parameters.
O05: two endpoints in different rounds never receive the exact same error accidentally
     from reused seeds. Independent per-event stream is tested.
O06: raw event serialized length128 and burst length160; byte accounting includes both.
O07: allcaps hold pathwise, including cal-only/combined/full actions and STOP. Distinguish
     decision count from execution count. No infinite zero-cost nonSTOP action.
O08: byte reference computed from public record schema before case/fault execution.
O09: X0 is identical across methods for a case; at zero budget outputs are equal.
O10: record compression/binning cannot inspect fields absent from public history.
O11: first sample does not encode a known common simulator start; recorder-arm/phase
     are hidden and independent of fault labels. No true-time sequence ordinal or
     synchronized start marker appears in public tensors. Preserve legitimate message
     matching and signal-based timing inference.

## Split isolation and representation

D01: no primary test graph is labeled-isomorphic to train/validation; node renaming alone
     rejected. WL hash collisions get exact checks.
D02: seen/clock/implementation suites intentionally share anchors, but case streams and
     relevant composite identities do not leak. Do not falsely reject intended pairing.
D03: target/test IDs, filename, fault class, offset, impl version and global tick are
     absent from model input; poison private metadata and verify outputs unchanged.
D04: random node/clock renaming remaps predictions and chosen actions equivariantly.
D05: no raw numerical message IDs; replacing IDs with a bijection preserves output.
D06: all branches/snapshots from a case share split and crossfit fold.
D07: primary test target count varies with graph; invalid target logits masked; valid
     probabilities sum to1; larger graph does not require reinitializing model layers.
D08: unseen connection matched type/count checks and type/relationship coverage reported.
D09: data validation catches same graph disguised by continuous parameter resampling.

## Learning and policy logic

L01: supervised q loss equals manually computed true-label -log2 probability.
L02: r_t telescopes exactly (numeric tolerance) to final-minus-initial log2 score.
L03: reward can be negative; no clamp-to-zero or discounted sum in the primary objective.
L04: labels/rewards only in trainer; forbidden in forward selector input during train/test.
L05: frozen q weights unchanged through policy updates and all evaluations.
L06: Bellman target terminal/bootstrap/STOP masks and horizon indices match hand fixtures.
L07: a delayed-payoff two-step fixture trains a value model differently from myopic rule;
     do not require success on a real held-out system for this unit test.
L08: GNN Plan-Once action sequence identical when intermediate returned records are altered;
     its FINAL shared-GNN diagnosis may change. A trained GNN selects the system-specific
     plan from the initial evidence and structure before new execution.
L09: Replan-Each-Round selectors can change actions with updated evidence; not a hard-coded schedule.
L10: Transformer Replan-Each-Round sees the same public graph/edge data and actions as
     GNN Replan-Each-Round; both use the same final GNN diagnoser.
L11: candidate ordering permutation changes score order only; no fixed action-index bias.
L12: same replay/source budget, update caps, frozen reader and validation effort across
     learned families; parameter counts and time disclosed.
L13: no test BatchNorm/statistics/temperature/early stopping updates.
L14: out-of-fold q score producer has not trained on that source topology.

## Evaluation and reporting

E01: exact toy checks and optional DP remain labeled restricted finite references.
E02: MuJoCo plotted metric is NLL/log-score/accuracy, not exact IG or claimed optimum.
E03: timeout/NaN/abstain remains in denominator with recorded fallback probability.
E04: paired intervals bootstrap system/config IDs then cases, not frame rows.
E05: every registered suite, method, seed and budget is present or explicitly failed.
E06: unrestricted full reference labeled infeasible whenever caps exceeded.
E07: repair/audit task streams disjoint; hidden audit result never affects another action.
E08: audit all20pass reported as a protocol outcome, not proof of risk below1%.
E09: independent reader reuses exactly the same evidence hashes without reacquisition.
E10: report includes prior baseline, target cardinality, actual costs, training overhead,
     and allheldout per-system results. No selection of only successful systems.
E11: selected checkpoint and every report table trace to artifact/config hashes.
E12: raw/source retention disclosed; reproducer can find stored manifest and scripts.

## Release completion

Formal outputs can be marked completed only when P0–P6 artifacts and test logs exist.
Separate `implemented`, `ran`, and `passed`. A `--help` success is not experiment execution.
A technical smoke run is explicitly excluded from final scientific estimates.
No expected positive numeric result is part of this checklist.
