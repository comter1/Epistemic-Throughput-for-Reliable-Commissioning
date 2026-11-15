# MAIN EXPERIMENT V2 — MuJoCo commissioning generalization

The active protocol is CGX-2.0. Read `docs/MAIN_EXPERIMENT_V2.md` and
`configs/protocol.json`. CGX-1.0 and the temporary single-seed P48 proposal are
superseded and archived in `docs/archive/pre_v2/`. Actual execution is reported
in `STATUS.md` and the external command/job records; implementation is not completion.

Train32×128, validation8×64, seen8×100, unseen16×100 and larger unseen8×100 cases
are fixed before training. All six main methods give their collected evidence to
the same final GNN diagnoser. Their names identify how that evidence is selected:

| Internal key | Public method name |
|---|---|
| `gnn_fixed_plan` | GNN Plan-Once + GNN Diagnoser |
| `gnn_adaptive` | GNN Replan-Each-Round + GNN Diagnoser |
| `transformer_adaptive` | Transformer Replan-Each-Round + GNN Diagnoser |
| `gnn_depth2` | GNN 2-Step Planner + GNN Diagnoser |
| `cost_aware_adaptation` | GNN Cost-Aware One-Step + GNN Diagnoser |
| `full_recording_feasible` | Full Recording + GNN Diagnoser |

GNN Plan-Once learns from source systems and creates a complete inspection plan
from each new system's structure, initial evidence and budget. It executes that
system-specific plan without revising it after intermediate observations. The two
Replan-Each-Round methods choose a new inspection after reading each observation;
their selectors use a GNN or Transformer, while their final diagnoser stays the same.
Full Recording uses a recording rule to select evidence and the shared learned GNN
to diagnose it. See [method descriptions](docs/METHODS.md) for all six methods,
the centralized display-name mapping and the separate independent-reader analysis.

The five methods with learned evidence selectors use three seeds.
B=1,2,4 counts additional diagnostic executions. Bytes,
calibrations and time are reported separately. Each model has a100k-update maximum
with validation every2000 updates and preregistered patience10; best validation
checkpoints, thresholds and preprocessing are frozen before testing.

The main matrix contains147,200 cells and6144 repair/audit conditions. Clock,
implementation and horizon sensitivity are optional appendix work. Real physical
failures, timeouts, abstentions and unfavorable results for any method remain in the results.
Legacy `finite_delay_benchmark_complete`, `finite-delay-plotting` and `result_review`
are read-only and checked by file hashes and their original tests.

## Running and monitoring

```bash
export CGX_DATA_ROOT=/home/leiyo/work/cgx_data_v2
export CGX_PYTHON=/home/leiyo/work/cgx_env/bin/python
export CGX_REMOTE_WORKERS_FILE="$CGX_DATA_ROOT/reports/remote_workers.json"
export CGX_EVAL_WORKERS=16 CGX_FEATURE_READERS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$CGX_PYTHON" -m cgx.cli report --data-root "$CGX_DATA_ROOT"
```

Follow `docs/06_IMPLEMENTATION_AND_RUNBOOK.md` for qualification and launch.
After the seal and compatible-source import, `bash scripts/run_main.sh` executes
the complete resumable pipeline. Each of the two local B200s and the B200 at
`leiyo@ssh.cloud.sdu.dk:2127` owns one independent training job. The remote host
uses the same shared filesystem. SSH private keys are outside the repository and
excluded from exported artifacts. No DDP, model enlargement or test-time fitting.

Inspect `main_run/status.json`, stage logs, `jobs/*/status.json`,
`jobs/*/remote_status.json`, `commands/*.json` and `reports/status.json` under the
external root. Sealed source, dependency, asset and manifest hashes are retained
in `software/` and `protocol_lock.json`. All source data reuse is independently
verified against old sealed inputs; new transitions and outputs have their own
V2 indices. `reports/resource_forecast.json` records measurements and assumptions.

Results include the four main figures, all per-case observations and resources,
paired configuration/case bootstrap intervals, seed variation, independent-reader
checks, and repair/audit outcomes. Learned log-score improvements are not exact
mutual information; twenty successful audit runs are not a risk certificate.

Original generated reports, figures and sealed source snapshots are retained with
legacy display names for provenance. Corrected presentation exports use the names
above and preserve the original result files, internal method keys and numerical
values. In particular, the historical label "Fixed plan" means the learned,
system-specific GNN Plan-Once method. See [historical-name guidance](docs/METHODS.md#historical-names-and-reproducibility).

## Exporting corrected method names from saved results

From this experiment directory, generate presentation files in a new or empty
output directory without running collection, training, evaluation or aggregation:

```bash
PYTHONPATH=src /home/leiyo/work/cgx_env/bin/python -m cgx.reporting.presentation \
  --data-root /work/Users/leiyo/cgx_data_v2 \
  --out /work/Users/leiyo/cgx_data_v2/presentation_method_names
```

The presentation output contains the corrected reports, tables and figure labels.
The original `results/` and `release/` directories remain preserved artifacts using
legacy display names. CSV presentation copies keep internal method keys and add
human-readable display names. The saved protocol and checkpoint hashes continue
to identify the original scientific run.
