# V2 implementation and runbook

This document supersedes the archived CGX-1.0 runbook. See `MAIN_EXPERIMENT_V2.md`
for the scientific plan. Source selection is deterministic and does not use outcomes.

1. Use the isolated environment, fetch/verify the pinned Panda asset and inspect all
   three GPU UUIDs. `doctor` requires the verified remote-worker registry as well as
   the two local devices. Keep a single trainer per GPU.
2. Set `CGX_REUSE_SOURCE=/home/leiyo/work/cgx_data` when generating the V2 manifests.
   `make-manifests` selects the first32/8 source topologies and balanced case hashes,
   generates all held-out topologies, executes healthy routing fixtures, and validates
   structural identity, paired type counts, family balance and disjoint RNG streams.
3. Execute `legacy-check`, `qualify-system`, `scripts/qualify_source.py`, pytest with
   `--junitxml=$CGX_DATA_ROOT/reports/pytest.xml`, and `tools/validate_handoff.py`.
4. Execute `scripts/technical_smoke.py --data-root "$CGX_DATA_ROOT" --updates 100
   --batch-size 128`. It collects real technical evidence, trains all five readers
   and all five policy families, creates OOF scores, validates and repairs/audits.
   Its1000 actual updates and all outputs are excluded from scientific tables.
5. Profile1000 representative physical executions and1000 optimizer updates using
   `cgx.env.qualification.profile_executions` and `cgx.training.profiling.profile_updates`.
   Run `scripts/profile_v2.py` on source-only cases to measure actual cases/s,
   transitions/s and evaluation cases/s including acquired-evidence storage. Keep
   serial/parallel equivalence evidence. Construct the resource forecast from these
   measurements using `cgx.runtime.forecast.resource_forecast`.
6. `lock-protocol` refuses incomplete prerequisites, failed/skipped tests or a smoke
   report from different source code. It seals manifests, config, source and assets.
7. `import-compatible-data --source-root /home/leiyo/work/cgx_data` verifies exact
   source graph, task, case, RNG, physical/observation/behavior contracts, marker and
   shard hashes. Source data stays preserved; unselected shards never enter V2 replay.
   Imported completed physical executions count toward the200k training-data cap.
8. `run-main --gpus 0,1 --workers 16 --resume` continues source/validation collection,
   strict validation, lossless feature preparation, five reader jobs, OOF scoring,
   fifteen policy jobs, validation-only scalar/checkpoint selection, freeze, all
   147,200 main evaluations, same-transcript independent rereading,6144 repairs/audits,
   paired aggregation, four figures and export. The remote registry automatically
   adds the third GPU. Optional appendix work is excluded from required completion.

The supplied shell wrapper requires CGX_DATA_ROOT and CGX_PYTHON. Use a detached
supervisor and save its PID, command and log path. A failed phase stops subsequent
phases; resume revalidates prerequisites and outputs. Do not relaunch while an
existing driver or recorded child is alive. No test results may amend training,
thresholds or the registered test cases. After a source change, preserve the old
seal and use an explicit versioned amendment; do not silently replace hashes.

Every stage distinguishes implemented, running, interrupted, failed and passed.
A forecast being computed successfully does not mean a scientific stage passed.
`STATUS.md` and the external root are the authority for actual completion.

## Presentation-only method-name updates

Use [METHODS.md](METHODS.md) for the current full display names. All six main methods
use the same final GNN diagnoser; the method name describes evidence collection.
Archived documents, original `results/`, original `release/` and sealed source
snapshots retain their legacy display names for provenance.

To export corrected labels and explanations from the saved numerical results, run
this separate presentation command from the experiment directory. The output
directory must be new or empty:

```bash
PYTHONPATH=src /home/leiyo/work/cgx_env/bin/python -m cgx.reporting.presentation \
  --data-root /work/Users/leiyo/cgx_data_v2 \
  --out /work/Users/leiyo/cgx_data_v2/presentation_method_names
```

This path does not rerun the scientific experiment or replace original result
files. It preserves internal method keys in CSV exports and adds display labels.
Use this command for naming changes instead of the scientific pipeline's
`aggregate` or `figures` commands, which belong to the sealed run. Presentation
exports do not replace the original protocol, source or checkpoint hashes.
