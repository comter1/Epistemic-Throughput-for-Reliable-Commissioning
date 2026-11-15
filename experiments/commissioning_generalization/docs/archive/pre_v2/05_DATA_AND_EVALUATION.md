# 05. Data collection, comparisons, statistics and engineering outcomes

## Source data

After sealing split manifests, create eight acquisition trajectories per source case:
two independently seeded trajectories at each nonzero byte fraction .1,.25,.5,1.
Mix behavior policies 40% random legal, 30% interface rules, 30% diverse fixed plans.
Sample execution caps from the common training distribution in protocol.json. Do not
use target labels or injected severity to choose actions. Truth is for scoring only.

Collect until all designated trajectories complete; target about 1–2M transitions, not
an exact quota obtained by duplicating examples. If early stopping produces fewer,
report actual size. Add a predeclared training-only branch coverage pass for weakly
covered action families, with total physical training executions <=2M. Every learner
gets the same resulting data and branch access. No method-dependent larger dataset.
All branches of a case/topology belong to the same source fold and split.

Use immutable history references rather than copying the growing history into every
transition. Store once per acquired execution; transitions reference it. Encoding caches
must be keyed by transcript hash AND encoder/predictor version and may not contain
unacquired fields. Save raw-wire cost before preprocessing.

Default raw retention: preserve all evaluation decision/evidence traces used for the
primary test and engineering evaluation, and a deterministic 2% stratified source subset.
Other source runs can retain lossless compressed raw data when storage permits, or
fixed preprocessing plus complete reproduction seeds/config hashes. State clearly if
raw source logs are pruned. Do not claim byte-for-byte reproducibility from an unpinned
simulator. Measure storage for first 1000 technical executions and forecast full volume.
This estimates capacity, not selects research conditions. Stop cleanly before disk-full.

## Comparators and fairness

Five learned families x five seeds: gnn_adaptive, transformer_adaptive, gnn_fixed_plan,
gnn_myopic, gnn_depth2. Cost-aware rule reuses myopic next-loss predictions and selects
its beta only on validation. Also interface_rule, default_only, full_recording_feasible
and explicitly infeasible full_recording_unrestricted_reference.

The hand rule uses public types and measured residuals: compare paired module/edge
records, match provenance, calibrate cross-clock timing pairs when unresolved, prioritize
suspect paths according to acquired evidence. It must not merely scan node indices or
look at hidden labels. Write down the complete rule and validation-selected thresholds.
It is allowed to be strong. A weak random rule is not the main baseline.

All policies receive the same graph, default evidence, action library, resource bounds,
nominal parameters, and shared final q. Topology information is given to the Transformer
and rules too. Cost-aware and depth-two learners use the same source cases and update
cap. Ground-truth probability-model planners are privileged references, not feasible
competitors in the data-only target-system setting.

Full-recording is benchmarked honestly. If it is cheap enough to win, report that. No
fabricated latency penalty, forced input truncation or weaker predictor just for full logs.
Unrestricted reference may exceed caps but cannot be counted as a feasible completion.

## Evaluation matrix

Main acquisition evaluation: all five suites, five byte fractions (including zero),
main execution cap4/calibration cap2. Source and held-out conditions use same target,
cost and timing definitions. Every case is paired across methods on latent fault,
initial evidence, task and exogenous noise streams. Each method's observation noise
is keyed by case/run/port/event, not by how many random numbers its code happened to draw.
Independent repetitions, not accidental reused random draws, are used across executions.

Policy weights and q are frozen. No target labels used for temperatures, normalization,
policy adaptation, reward feedback, hyperparameters, early stopping or offline branch
queries. BatchNorm (if present) must use source-frozen statistics; preferably LayerNorm.

At byte0 all acquisition policies STOP and must agree with q(X0), within numeric tolerance.
If they differ, implementation is invalid. For every budget report outcomes for EVERY
case including timeouts, abstentions, healthy cases and visibly truncated records.

## Metrics

Primary: mean final joint-label log-loss in bits on test_connections at byte fraction .25.
Primary comparison: paired gnn_adaptive versus transformer_adaptive. Also required strong
rule and fixed-plan comparisons. Positive means are not assumed. Do not substitute a
favorable metric if the preregistered one is weak.

Additional: initial and final log-loss; true-label log-score improvement; joint location+
mechanism top-1 accuracy; mechanism-only accuracy; healthy false-alarm rate; confusion
by mechanism, topology size and severity; timeout/abstention rate; bytes, executions,
calibration bursts, decisions, actual wall time, CPU/GPU selection/inference time.
Report fixed-prior baseline loss per graph so target-space size changes are visible.
Never label q entropy reduction, calibration confidence or accuracy/cost as exact ET.

For cost-to-quality plots, use a validation-chosen target level and evaluate only recorded
budget points. If a method never reaches it, mark “not reached”; do not extrapolate an
invented crossing. Fixed per-case caps and average spent-cost frontiers are different
plots and must be labeled accordingly.

## Statistical unit

Average cases within configuration, then configurations equally within suite, then
training seeds equally. Report each seed separately and mean/spread. Paired bootstrap
resamples configurations with replacement and cases within them, with the SAME indices
for all methods (2000 replicates). Model-seed uncertainty is a separate axis; optionally
hierarchical resampling across seeds can be added and labeled. Repeated observations or
frames are not independent cases. Use a single preregistered primary contrast; apply
Holm correction to predeclared secondary significance claims. Do not turn exploratory
subgroup selection into a confirmed headline.

## Feedback value and topology value are separate

Main simulator comparison of gnn_adaptive vs trained gnn_fixed_plan measures achieved
feedback advantage in these policy classes, not the exact value of adaptivity. Only
finite fixtures with both optima computed can claim optimal-adaptive minus optimal-fixed.
Main unseen-topology result measures frozen performance on held-out connections, not
simply acceptance of a variable-length graph. Include all held-out graph IDs in reports.

## Independent reader

Re-score the identical acquired transcripts with the independent source-trained reader.
No new acquisition and no new test labels for learning. This identifies reader dependence,
not true MI. Report both readers even if the primary result changes.

## Repair and acceptance

Preregister a deterministic subset: four configurations in each test suite, selected by
hash BEFORE model training; 64 cases/config, balanced 16 per family, selected the same way.
Main byte fraction .25; four methods (gnn_adaptive, transformer_adaptive, interface_rule,
gnn_fixed_plan); learned seeds 11,23,37. Nonlearned rule results need not be rerun for
identical seeds; reuse paired cases and label this dependence.

Choose confidence threshold from {.5,.7,.8,.9,.95,.99} on validation to maximize accepted
successful cases subject to observed validation accepted-failure <=.05; if none qualifies,
abstain. Freeze the selected threshold and show full risk–coverage curves too. This is
not a formal probability certificate, especially under shift.

Given the final prediction, execute the corresponding ordinary node repair or healthy
no-change, or abstain. Remove temporary probes before deployment verification. Two
fresh public-verification executions must satisfy declared requirements to declare
completion. Reserve these two executions outside diagnostic cap and include them in
total deployed cost; there is no free unlimited re-testing until passing. Abort/timeout
is incomplete. No repair/retry based on hidden audit observations.

External evaluator runs 20 fresh audit executions for each declared completion, using
new disjoint task/disturbance streams. W=1 means all20 pass. Z=1 is declared completion.
Report coverage=P(Z=1), risk=P(W=0|Z=1), yield=P(Z=1,W=1), per-execution failure rate and
unnecessary modifications to healthy modules. Risk is undefined at zero coverage.
The all20-pass protocol is NOT a certificate that deployment failure probability <1%.
Training and external audit compute are reported separately from deployed acquisition cost.

## Figure and table deliverables

F1: final NLL/accuracy vs actual byte budget, seen and unseen connections, same axes.
F2: held-out system paired cost/quality differences, all systems rather than only winners.
F3: learned closed-loop vs trained fixed-plan, with costs/rounds reported.
F4: topology, clock, size and implementation suite comparison, all budgets in supplement.
F5: repair risk–coverage and yield–total-cost, independent audit protocol labeled.
F6: measured selection latency, training/source data cost, parameter counts, memory.

Keep the original nine-panel finite benchmark untouched. New outputs cannot overwrite
original figures or CSVs. Do not copy their exact-ET y-label into the MuJoCo figures.
Release per-case results, trace index, job status, configuration/provenance hashes,
paired statistics and the actual selected checkpoint for every run.
