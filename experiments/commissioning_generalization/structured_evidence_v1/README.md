# Structured Evidence Diagnoser on the existing Panda benchmark

This namespace implements the requested reader replacement and a full paired
rerun of the existing commissioning experiment. It does not change the simulator,
topology generator, fault families, manifests, policy architectures, training seeds,
budgets, or main evaluation matrix. The existing `src/cgx` implementation remains
the reference for those components. New artifacts go to a separate run directory;
`OverlayStore` reads original public evidence and writes only new evidence.

The experiment has 3,200 physical test cases and 147,200 method/seed/budget cells.
The primary unseen-connections suite has 1,600 cases, evaluated with three policy
seeds per learned method and budget. These repeated policy runs are not independent
physical cases.

## Diagnostic model

An opaque acquisition ID and a measurement event ID distinguish a new measurement
from another storage appearance. Identity is assigned once before packaging changes.
Exact event IDs deduplicate; numeric equality does not. IDs enter only equality and
grouping operations, never numerical fault features. Conflicting values for one ID
raise an error. Provenance has a separate content address from raw observation data,
because identical values alone cannot establish whether two events are identical.

For timing, the public graph defines token-preserving paths. The model is

`timestamp = message_epoch + persistent_clock_offset + fault_delay × downstream + residual`.

Message epochs are integrated by weighted centering with their contrast covariance.
One Gaussian clock offset is shared by all acquisitions in the same clock group.
One fault magnitude is shared across a case. Calibration contributes observed
four-timestamp clock-difference factors. Signal factors compare acquired adapter
input/output packets using the public nominal transfer relation, integrating a
shared gain error. The fault posterior is a normalized sum over the original legal
healthy/component/family targets, with the original probability floor.

The original TRAIN measurements fit healthy timing and signal noise, calibration
noise, clock variance, three-component delay mixtures, and a six-component gain
mixture. The final reader and three original topology-exclusion readers fit
separately; even noise and clock parameters obey the exclusion. No private clock
offsets or injected severities enter fitting or inference. Fault labels enter
source fitting and evaluator metrics only. There is no test-time learning.

This first model has explicit limits: external motion records are retained but
their likelihood is fault-independent, so an external-only history returns the
legal prior. Missing values and timeout status alone are uninformative. Conditional
timing residuals are independent after the shared message, clock, and fault
variables; additional correlated downstream packet-age residuals are not separately
latent. These choices are fixed before held-out reader results are examined.

Acquisition counts, chosen actions, remaining budget, and storage length have no
direct likelihood for the fault. A genuinely independent event may legitimately
add information. The selectors retain their original state representation and may
use resource or coverage information to choose measurements.

## Ordered execution

Run commands from `experiments/commissioning_generalization` with `PYTHONPATH=.:src`.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for process pools.
Use the run's recorded environment snapshots and command files for exact paths and
hardware allocations.

1. `prepare` seals the unchanged benchmark and hashes original artifacts.
   `record_source_amendment` documents the earlier reporting/display-name changes;
   it does not rewrite or bypass the original experiment's protocol lock.
2. `fit_reader extract` reads immutable TRAIN replay, then `fit_reader fit` fits
   `final`, `crossfit_0`, `crossfit_1`, and `crossfit_2`. Run reader controls and
   source-fold audits, then `gates seal-parameters`.
3. Verify cached and uncached reader computation, then `gates authorize-replay`.
   `replay inventory` and sharded `replay worker` calls diagnose exactly the saved
   old histories. Baseline-only GNN/R-C work can run independently before the new
   reader is sealed. Their deployed deadline/fallback outputs are retained as a
   separate comparison from untimed native GNN rereads.
4. `invariance transformations` checks duplicates, real split/merged files, action
   metadata, identity, and shared-clock uncertainty on fixed physical histories.
   `replay finish` initially writes `predictions_complete`; `invariance finish`
   evaluates genuinely acquired execution increments; rerunning `replay finish`
   then writes the final replay completion marker.
5. `policy_training score` regenerates every q and reward using the new OOF
   readers. This preprocessing can overlap fixed-reader replay after reader
   sealing. `gates authorize-policies` requires complete replay and invariance
   before any selector optimizer step. Train all five original networks at all
   three seeds with `policy_training fit`, unchanged optimization and source-only
   validation selection. `dispatch` coordinates independent GPU jobs.
   Where measured utilization leaves GPU compute idle, `dispatch_shared_gpu`
   can co-schedule independent fits belonging to this same run. It verifies the
   occupants and memory headroom first. Each fit retains its own original RNG,
   optimizer, batches, and stopping rule; allocations and validation pool sizes
   are recorded explicitly. `dispatch_cpu` distributes the later main evaluation
   without requiring an idle GPU on that host.
6. `evaluation select-beta` uses the original validation grid. `evaluation freeze`
   seals all selected models and scientific runtime files. Sharded
   `evaluation evaluate` runs the original main matrix, followed by `evaluation
   finish`. Failed/timeout cells retain fallback predictions and remain counted.
7. After replay is complete, `reporting replay_reports` and `reporting
   replay_policy_contrasts` produce the fixed-history tables and paired intervals.
   `error_audit --cohort old_histories` describes modeled evidence in all wrong
   structured-reader replay contexts. This analysis can overlap selector training;
   completed replay and audit artifacts are preserved instead of rerun.
8. After `evaluation finish` confirms all 147,200 new main cells, run `reporting
   main_reports`, the predeclared `runtime_audit`, and `error_audit --cohort
   new_main`. These independent analyses retain the complete classification
   denominators. Both error-audit cohorts and the reader-deadline audit must be
   complete before `reporting interpret`.
9. Run `evidence_export old_histories` and `evidence_export new_main` to write
   `evidence_export/<cohort>` with packed histories, acquisition
   identities, traces and dependency-resolution checks. Raw measurement blobs
   stay at the recorded external locations; the analysis ZIP includes their
   resolution index and does not claim to embed those values.
10. `reporting interpret` and `plots` produce the eight requested interpretations
    and eight figure families from the completed tables and audits. Figure 05
    separately shows original deployed predictions, the structured reader on
    identical old histories, and retrained policies with the structured reader.
    `finalize audit` then verifies preservation and all required completion
    markers; `finalize package` creates the analysis ZIP with per-case
    probabilities, evidence references and code.

A SOURCE-only cold-call profile found a large deferred diagnosis slightly above
the original two-second deadline. Therefore, before any new main outcomes, a
separate `runtime_audit` was declared: after the official main matrix completes,
reread only diagnostic-timeout final histories without a reader deadline, using
the frozen model and exact saved evidence. Official results remain unchanged;
unreplayable cells retain their official prediction. This sensitivity analysis
cannot recover actions abandoned after an earlier timeout.

The physical simulation cache reuses an exactly identified noisy realization,
not a different measurement. Logical executions, bytes, and calibrations retain
their original accounting. Actual new simulation attempts, failures, cache hits,
selector time, diagnosis time, and wall time are reported separately. The original
two-second selection and diagnosis deadlines remain in the deployed evaluation.

## Evidence and interpretation

Reader-only comparisons hold the acquired history fixed. Comparing old actions
with the new reader against retrained actions with that same reader measures the
combined acquisition, retraining, and runtime change. It does not uniquely label
every remaining error's cause. The model's identity invariance is separate from
its diagnostic accuracy.

Shared-error tests distinguish the uncertainty of an uncalibrated transit mean
from the conditional posterior of identifiable clock offsets. The former retains
a shared-clock term; the latter can legitimately contract when the data identify
the offset. The independent-clock curve is explicitly a debug reference.

Metrics use all-case accuracy and fault-only joint accuracy, conditional healthy
false alarms and fault misses, true-target NLL in bits, multiclass Brier score,
and ten-bin ECE. Paired intervals average policy seeds within physical cases,
then bootstrap configurations and fault-stratified cases 2,000 times. Raw counts
accompany percentages. The same previously studied test manifests are deliberately
reused for paired comparison; this is not an untouched new test set.

The run retains raw acquired histories, event provenance, selected checkpoints,
source hashes, command logs, and all per-case predictions. The delivery ZIP includes
the analysis and content-address references; large raw stores and checkpoints stay
in the full run directory and are not silently discarded.
