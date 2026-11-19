# Rigid payload commissioning

CGX-PAYLOAD-COMMISSIONING-V1 compares one original task, fixed multi-pose checks,
fixed pose-and-motion checks, and evidence-dependent checks after a rigid payload
change. All methods use the same estimator, controller, action library and actual
motion acceptance. Existing-settings and privileged true-settings references are
reported separately. This is simulation, with no real-robot control interface.

The implementation is isolated from the previous software-fault experiments.
The legacy position actuators do not provide an independently configurable load
model, so this experiment uses a separately compiled torque controller model.
The physical plant owns the unknown load. Estimation uses measured encoders,
simulated execution-torque sensors and a third, public dynamics model.

For a fresh output directory, run from `experiments/commissioning_generalization`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:src CGX_DATA_ROOT=/work/Users/leiyo/cgx_data_v2 \
/home/leiyo/work/cgx_env/bin/python -m payload_change_v1.prepare \
  --run-all --run-root /ABSOLUTE/EXTERNAL/RUN_ROOT
```

The dispatcher must run on a host with SSH access to the three authorized nodes
at ports 2379, 2982 and 2404. Physics and bulk estimation run on those nodes with
40, 80 and 160 workers; fewer cases use fewer workers. GPU computation is not
needed. The pinned Panda asset and external data mount must be accessible on all
nodes. The tested environment uses MuJoCo 4.0.1; exact packages are recorded per
node. No environment upgrades are performed.

`prepare` writes the frozen controller/library/workload configuration and records
20 source profiling blocks remotely before invoking all stages. It is a replication
convenience added after the original run; the original preparation commands are
preserved in that run's command log. To resume an already prepared directory, use
`python -m payload_change_v1.run --stage all --run-root ABSOLUTE_ROOT`, or name one
stage. Preparation refuses to modify sealed runs.

Stages are `audit`, `qualify-source`, `fit`, `validate`, `seal`, `evaluate`,
`report` and `package`. Every stage records inputs, command, completion status
and exit code. The physics ledger reserves time before stepping and conservatively
charges interrupted attempts. The cap is 100,000 ten-second execution equivalents,
including source, qualification, failed attempts, verification and private audits.

Source has 192 physical cases, validation 64, and test 192 with three paired noise
seeds. The test matrix contains 3,456 runs and eight actual private tasks per run.
Validation chooses one stopping threshold shared by all four main methods. Test
outcomes cannot change parameters or cases. Confidence regions are approximate
and empirically calibrated on source; their held-out coverage is reported.

The original L0 retains its exact legacy 100 ms reference transitions. Those can
fail the declared limits even when the independently specified smooth workload
passes. These outcomes must remain distinct. All source controller candidates,
failed qualification trajectories and corrections are retained in the run archive.

Start analysis with the run's `FINDINGS_CN.md`, `CONTROLLER_DECISION.md`,
`protocol_locked.yaml` and `results/summary.csv`. The analysis ZIP is strictly
smaller than 500,000,000 bytes. Full 100 Hz sensor/software records and 500 Hz
private trajectories remain in the external raw store with SHA256 references;
representative traces and complete per-run summaries accompany the analysis.
