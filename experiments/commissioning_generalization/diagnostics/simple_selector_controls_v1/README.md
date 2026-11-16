# Simple selector controls, CGX-SIMPLE-SELECTORS-V1

This isolated extension asks whether a learned inspection selector is needed when
all methods already use the same fitted and frozen Structured Evidence diagnoser.
It does not train networks, refit the reader, or change the original task, actions,
cases, execution/calibration/decision/byte limits, or measurement identities.
Original `structured_evidence_v1` code and run artifacts remain read-only.

The six controls are one full execution, one full execution with up to two
preplanned calibrations, uniformly random affordable action keys, forward graph
coverage, a validation-selected fixed template, and the original interface-rule
formula with validation-selected stopping. The shared diagnoser still contains
source-fitted parameters; “no selector training” does not mean no learning anywhere.

The eight templates and three stopping options are evaluated on the original 512
validation cases at each of B=1/2/4: 16,896 logical rows. Selection uses
configuration-balanced **deployed** final NLL, then bytes, executions,
calibrations, and a fixed candidate order. Completed-reading NLL is reported but
cannot change the selection criterion. `selection_manifest.json` seals choices
and runtime hashes before opening new test results.

The new test matrix has 64,000 logical rows over 3,200 cases, with 32,000 primary
unseen-connection rows reported before the two transfer suites. Deterministic
methods have a null seed. Random seeds 11/23/37 are repeated choices on the same
cases. The five existing learned selectors and four-execution full-recording
reference contribute 147,200 imported rows. One-full-execution controls run at B=1
only; B=2/4 display points reuse a verified feasible B=1 trial and are not new trials.

Online evaluation preserves the original two-second selection/diagnosis deadlines.
Completed reading only finishes diagnosis of the exact saved history/provenance,
with a registered 120-second crash watchdog. It acquires nothing and cannot recover
checks lost to early termination. Fallbacks, failures and unresolved rereads remain
in all denominators. Imported measurements and probability vectors are reused
after integrity checks, not rerun to generate favorable timings.

Run commands below from `experiments/commissioning_generalization` with
`PYTHONPATH=.:src` and BLAS/OpenMP threads limited to one. Actual simulation,
diagnosis, statistics and packaging should run on the authorized remote CPU nodes;
the local machine coordinates. The dispatcher records exact SSH commands and
configured workers. GPU training is unnecessary for this experiment.

Core pipeline entry points are below. `SOURCE`, `SED` and `RUN` are placeholders
for the original source store, frozen preceding Structured Evidence run, and new
timestamped output directory. Use the exact recorded commands in this run's
`commands/` and `dispatch/` to reproduce its host paths and worker allocation.
Inventory/seals preserve existing files and do not silently overwrite an old run.
The source-only 32-case throughput pilot is a deterministic validation subset;
its measured throughput informs worker allocation, and its cells are retained in
the full 512-case validation matrix.

```bash
python -m diagnostics.simple_selector_controls_v1.prepare inventory --source-root SOURCE --sed-run SED --run-dir RUN
python -m diagnostics.simple_selector_controls_v1.import_previous --source-root SOURCE --sed-run SED --run-dir RUN
python -m diagnostics.simple_selector_controls_v1.test_policies --output RUN/tests/policy_controls.json
python -m diagnostics.simple_selector_controls_v1.test_runner --source-root SOURCE --sed-run SED --run-dir RUN
python -m diagnostics.simple_selector_controls_v1.prepare seal-code --run-dir RUN
# RUN/validation/pilot_cases.json contains the preselected source-only 32-case IDs.
python -m diagnostics.simple_selector_controls_v1.dispatch --source-root SOURCE --sed-run SED --run-dir RUN --phase validation --pilot
python -m diagnostics.simple_selector_controls_v1.dispatch --source-root SOURCE --sed-run SED --run-dir RUN --phase validation
python -m diagnostics.simple_selector_controls_v1.evaluation finish --source-root SOURCE --sed-run SED --run-dir RUN --phase validation
python -m diagnostics.simple_selector_controls_v1.selection --run-dir RUN
python -m diagnostics.simple_selector_controls_v1.dispatch --source-root SOURCE --sed-run SED --run-dir RUN --phase primary
python -m diagnostics.simple_selector_controls_v1.evaluation finish --source-root SOURCE --sed-run SED --run-dir RUN --phase primary
python -m diagnostics.simple_selector_controls_v1.reporting --run-dir RUN --phase primary --workers 8
# Read and preserve the completed primary report before launching transfer.
python -m diagnostics.simple_selector_controls_v1.dispatch --source-root SOURCE --sed-run SED --run-dir RUN --phase transfer
python -m diagnostics.simple_selector_controls_v1.evaluation finish --source-root SOURCE --sed-run SED --run-dir RUN --phase transfer
python -m diagnostics.simple_selector_controls_v1.evaluation finish --source-root SOURCE --sed-run SED --run-dir RUN --phase all
```

Reporting, figures and final delivery:

```bash
python -m diagnostics.simple_selector_controls_v1.reporting --run-dir RUN --engineering-checks
python -m diagnostics.simple_selector_controls_v1.reporting --run-dir RUN --phase primary --workers 32
python -m diagnostics.simple_selector_controls_v1.figures --run-dir RUN --phase primary
# Only after all transfer rows and the all-phase marker exist:
python -m diagnostics.simple_selector_controls_v1.reporting --run-dir RUN --phase all --workers 32
python -m diagnostics.simple_selector_controls_v1.figures --run-dir RUN --phase all
python -m diagnostics.simple_selector_controls_v1.finalize all --run-dir RUN --workers 32
```

The main paired comparisons at primary B=4 are GNN/Transformer Replan against the
selected fixed plan and graph rule. Each method first averages its seeds within
physical case; seed numbers are not paired between random and learned methods.
The 2,000-replicate bootstrap resamples configurations and then cases within each
configuration/fault-family stratum. Differences are **simple minus learned**.
Practical comparability requires accuracy's individual one-sided 95% lower bound
to be at least −0.01 and miss rate's individual upper bound at most +0.01. These
are declared tolerances, not equality inferred from a large p-value, and no joint
or simultaneous significance claim is made. Parallel report workers leave the
statistics and random seeds unchanged.

Bytes, executions, calibrations, selection time and diagnosis time remain separate.
Fresh physical work is counted before execution in a durable global ledger,
including qualification, validation and failed/interrupted attempts. Aliases are
deduplicated by `execution_source_key` when attributing actual compute. Cache hits
retain the original logical acquisition charges. Cross-run wall times are
descriptive because machine load, cache state and concurrency differ.

Reporting requires complete phase/import markers and their artifact hashes.
The four figure families retain the preceding study's learned-method colors and
include standalone PGFPlots sources, exact CSV data and vector PDFs. When LaTeX is
absent, PDFs use Matplotlib and compilation status is explicit. No scientific
conclusion is written from engineering fixtures or incomplete test matrices.

Finalization checks protected inputs, runtime seals, matrix sizes, tests, report
hashes and evidence resolution, and creates one ZIP strictly below 500,000,000
bytes. It includes every new/imported online/completed probability vector,
validation choices, source/diff, tables, figures, commands and integrity indexes.
The delivery uses `reports/normalized_analysis_compact.parquet`: an exact column
projection retaining all 422,400 rows, full target/probability vectors, metrics,
keys, logical costs, errors, action summaries and reader/evidence hashes. The four
original-format prediction tables remain included. The larger immutable full
normalized table stays outside the ZIP; its source hash, omitted columns and
row-for-row vector equality are documented in `provenance/compact_analysis_view.json`.
`finalize all` derives this view before auditing and packaging; separate stages
can use `finalize compact-view`, `finalize audit` and `finalize package`. No source
report or scientific table is overwritten by compaction. The implementation patch
always uses the pre-extension commit in `source_inventory.json` as its base,
including after packaging-only code amendments have been committed.
Large raw measurements, simulator caches, individual cell envelopes and checkpoints
stay in the existing storage with exact paths/content/file hashes. The archive is
sufficient for analyzing predictions; full replay still requires those stores.
