# 06. Implementation layout and required command interface

## Directory contract

All new code is under `experiments/commissioning_generalization/`:

```
AGENTS.md                 scoped agent rules
README.md, STATUS.md      implementation and executed status
docs/, configs/           frozen protocol and amendments
schemas/, contracts/      public/private APIs and serialized records
src/cgx/
  cli.py
  config.py, seeds.py, provenance.py
  graphs/                 typed generator, isomorphism, split manifests
  env/                    MuJoCo wrapper, packets, modules, controller, faults, repairs
  observation/            clock exchange, legal actions, recorder, budgets, encoding
  data/                   public views, private store, trajectory writer, cross-fitting
  models/                 reader, GNN Q, relation Transformer, shared candidate heads
  training/               diagnostic, fitted Q, fixed-plan training, checkpoints
  baselines/              rules, myopic, depth2, cost-aware, feasible full
  evaluation/             frozen acquisition, repair/audit, metrics, statistics
  reporting/              plots, tables, trace inspection
  runtime/                task queue, hardware, profiling, resume
scripts/                  local entry scripts, no implicit privileged changes
tests/                    unit + simulator integration + leakage + main-pipeline smoke
third_party/              local ignored pinned asset checkout and license copy
reports/                  small checked-in evidence summaries, not raw data
```

Use an isolated editable package named `cgx`. Keep generic `benchmark.py` imports out
of global namespace. Legacy tests run in their own directory with unchanged sys.path.
Generated corpora, embeddings, model checkpoints and long logs live under CGX_DATA_ROOT,
not in Git. Each run gets a unique protocol/config hash and an atomic completion marker.

## Dependencies and environment

Preferred Python3.11 or3.12 in a fresh venv/conda env within user-authorized storage.
Reuse an existing compatible PyTorch/CUDA runtime if it passes the hardware checks.
Use a B200/SM100-capable wheel and a host driver compatible with that wheel. Do not
hard-code an old CUDA12.8 instruction as “latest”, auto-upgrade host drivers, or install
into the global environment. Pin actual tested versions after smoke tests. See sources.

Expected dependencies: numpy, scipy, networkx, pandas, pyarrow, matplotlib, torch,
mujoco, pytest, jsonschema, psutil, and a lossless compression package if used.
No torch-geometric/custom CUDA scatter requirement for v1. No LLM API dependencies.
Save pip freeze, torch/CUDA/driver/device data, Menagerie commit+license, Python, CPU,
OS, thread settings and complete CLI configuration. Keep network-download steps explicit.

## Required CLI (to be implemented, NOT supplied as completed code)

From this directory after editable install:

```
python -m cgx.cli doctor --config configs/protocol.json
python -m cgx.cli legacy-check --repo <absolute-repo-path>
python -m cgx.cli qualify-system --source-fixtures --report reports/qualification.json
python -m cgx.cli make-manifests --config configs/protocol.json
python -m cgx.cli validate-manifests --strict
python -m cgx.cli lock-protocol
python -m cgx.cli collect --split train --resume
python -m cgx.cli validate-data --split train --strict
python -m cgx.cli fit-readers --crossfit --final --independent --resume
python -m cgx.cli score-transitions --crossfit-only --resume
python -m cgx.cli fit-policy --method gnn_adaptive --seed 11 --resume
python -m cgx.cli build-job-manifest --phase policies --out <data-root>/jobs/policies.jsonl
python -m cgx.cli run-queue --gpus 0,1 --jobs <data-root>/jobs/policies.jsonl --resume
python -m cgx.cli select-checkpoints --validation-only
python -m cgx.cli freeze-models
python -m cgx.cli evaluate --suite all --resume
python -m cgx.cli evaluate-round-sensitivity --resume
python -m cgx.cli reread --reader independent --same-transcripts
python -m cgx.cli evaluate-repairs --preregistered-subset --resume
python -m cgx.cli aggregate --paired --bootstrap 2000
python -m cgx.cli report --include-negative-results
python -m cgx.cli export --out <data-root>/release/cgx_main_results.zip
```

These commands MUST fail with a clear error if prerequisite manifests/checkpoints are
absent or their hashes differ. A CLI command returning exit0 without doing the promised
work is a test failure. `--resume` skips only an artifact with matching hashes and a
successful completion marker, not a directory that merely exists.

## Technical execution order

P0: snapshot legacy, branch and files, install isolated deps, inspect assets and hardware.
P1: real modules and fixed controller; healthy/fault/repair/missing-log tests pass.
P2: generate manifests, audit isomorphism/matching/leakage, record healthy qualification,
    seal all identities and source/test file access boundaries.
P3: collect source traces with random-stream separation and cost accounting, create folds.
P4: fit3 crossfit readers + final common reader + independent reader. Score transitions
    out-of-fold; fit all policy families with identical data and update budgets.
P5: validation-only selection; freeze checkpoint and preprocessing hash. Evaluate ALL
    registered suites; do not tune after failures or quietly replace a difficult graph.
P6: repair/audit subset, reporting, source/compute overhead and limitation disclosure.

Run one tiny end-to-end fixture for software correctness before formal data generation.
It must be flagged `technical_smoke=true`, excluded from every scientific table, and
cannot be cited as support for the hypothesis. This is not a new research pilot.

## On-device parallelism

Queue one training job per physical B200 by UUID. A job launched with CUDA_VISIBLE_DEVICES
has local device0; do not mistakenly call local cuda:1 when only one card is visible.
Use file locks and atomic job status transitions. Store success/failure, exit code, peak
memory and elapsed time. Failure does not silently count as another seed. Check other
users' jobs; never kill a process not owned by this experiment.

CPU simulator workers are independent processes with OMP/MKL/OPENBLAS threads=1. Respect
allocated CPU affinity/cgroup, not the host's advertised total cores. Start with min(16,
allocated_cores//2), at least1. Different workers have deterministic noncolliding seeds.
Do not launch a new pool for every minibatch. Prefer replay preprocessing and batched
inference to tiny GPU calls from thousands of CPU processes.

## Checkpoint/resume details

Store weights, optimizer, target network, scheduler, iteration, random states, replay
manifest, reader/version hashes, chosen protocol and validation history. Replaying a
resumed run must not silently regenerate split/label/action maps. Store target label
mapping by graph schema, not global padded indices whose order changes unpredictably.
Validation logs are append-only; negative results remain in the report.

## Artifacts and publication-sized export

Per-case table should include suite/configuration/case hashes, method/seed/checkpoint,
initial/final true-label NLL, top1 target, abstention, resource use, timing, errors,
acceptance and audit metrics where available. Labels belong in evaluator output, not
policy input. Save a separate decision trace with chosen action, acquired record hashes,
q distributions, budgets before/after and reason for STOP/timeout. Do not save true fault
in the public history cache. Export tables/plots/configs/reproduction scripts; checkpoint
weights and raw logs may be separate due to size. Original repository remains unchanged.
