# Codex task: implement the frozen main experiment

## Authority and scope

Implement protocol CGX-1.0 in the user's existing repository
`comter1/Epistemic-Throughput-for-Reliable-Commissioning`.
Use the local checkout supplied by the user; do not replace it with a fresh clone.
The uploaded snapshot inspected for this specification contains
`finite_delay_benchmark_complete/`, `finite-delay-plotting/`, and `result_review/`.
It contains no MuJoCo generalization experiment and no repository AGENTS.md.
The live checkout may be newer: inspect its actual instructions and status first.

Create branch **feat/commissioning-generalization-main** from the current authorized HEAD.
Write the new experiment under **experiments/commissioning_generalization/**.
Read existing root/ancestor instructions before installing the nested AGENTS.md.
The nested AGENTS.md applies only to the new experiment; never overwrite root instructions.

Do not reset, clean, force checkout, stash user changes, rewrite history, force push,
or alter git remotes. If tracked user changes make safe branching ambiguous, stop
and report the exact paths; do not discard them. If the target branch already exists,
inspect it rather than replacing it. Branch creation is required; commits and remote
pushes are not automatically authorized by this package.

`bootstrap_branch.sh /path/to/repo` previews this operation.
`bootstrap_branch.sh /path/to/repo --apply` creates a new local branch and copies the
nested files, but only when the destination and branch do not already exist and
tracked files are clean. It never installs packages or starts GPU jobs.

## Required reading order

1. `START_HERE_CN.md` in this handoff.
2. Nested `AGENTS.md`, `docs/00_PROTOCOL.md`, and `configs/protocol.json`.
3. All eight implementation documents (01–08), then contracts and schemas.
4. Existing finite benchmark's README, model, tests, sweep and archived settings.
5. `docs/09_SOURCES_AND_DECISIONS.md` for software compatibility and provenance.

Configuration values in protocol.json are authoritative. A conflict is an error
that must be resolved in a logged amendment, not by silently taking convenient values.

## Implement, do not propose another study

Deliver a real headless MuJoCo experiment, real executable cache/queue/adapter
faults, dataset creation, diagnostic training, finite-horizon action-value learning,
strong comparators, frozen-model held-out-graph evaluation, and repair/audit evaluation.
The finite numerical toy is a regression reference, not a substitute for these tasks.
Do not turn the assignment into a notebook showing a few hand-picked trajectories.
Do not add an LLM, train a robot policy, redesign the existing toy, or introduce a
new theorem as a prerequisite.

Technical qualification is mandatory: healthy controllers must run; faults must
affect the executed path; rewards and budgets must be correct. Qualification may
fix software bugs, not choose conditions because our method wins. Once manifests
and the protocol lock are written, all scientific changes require a new version.

## Execution sequence

P0: repository snapshot, safe branch, isolated environment, hardware report.
P1: typed executable system graph, fixed controller, physical fault and repair tests.
P2: graph generation and frozen splits; instrument/clock/budget and privacy tests.
P3: training-only trajectories, cross-fitted diagnostic rewards, action coverage report.
P4: common final predictor and all five policy families, five seeds, equal update cap.
P5: checkpoint selection on validation only; seal test access; five test suites.
P6: independent repair/audit subset, paired statistics, figures and machine-readable tables.

Complete each technical phase and continue to the next. Do not demand favorable
research results to proceed. Do not run expensive jobs until prerequisite artifacts
and checks exist. Prefer a resumable local queue so a disconnected SSH session does
not lose work. Respect any existing user scheduler; do not kill unrelated processes.

## Evidence of completion

Create `STATUS.md`, `IMPLEMENTATION_REPORT.md`, `reports/acceptance.json`,
`reports/environment.json`, `reports/protocol_lock.json`,
`reports/legacy_integrity.json`, and the outputs in the runbook.
Record actual commands, timestamps, git commit/dirty patch, hashes, device mapping,
CPU allocation, per-job states, durations, peak memory, and exit status.
Mark each item `not_started`, `implemented_not_run`, `passed`, `failed`, or `blocked`.
Never infer a passed test from a successful import. Never fabricate experimental values.
If access to packages, GPU, assets, or time is unavailable, deliver implemented code
with the precise blocked command; do not label the main experiment executed.

## Completion is not a positive scientific conclusion

An honestly negative result with correct implementation and complete evaluation is
an acceptable main-experiment outcome. Do not tune on held-out graphs, remove hard
fault cases, weaken baselines, or call learned log-score improvement exact ET.
All methods must see the same public graph, action catalog, evidence and resource caps.
