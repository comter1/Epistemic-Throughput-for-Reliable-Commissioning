# Codex task — MAIN EXPERIMENT V2

Continue the formal MuJoCo frozen-policy cross-topology experiment in
`feat/commissioning-generalization-main`, entirely within this experiment directory.
Read `MAIN_EXPERIMENT_V2.md`, `../configs/protocol.json` and the V2 runbook.
The original CODEX_TASK and single-seed P48 proposal are archived and superseded.

Use the public method names and explanations in [METHODS.md](METHODS.md). All six
main methods share the same final GNN diagnoser. GNN Plan-Once + GNN Diagnoser learns
to generate a complete inspection plan for each new system before new evidence is
collected; it does not revise that plan midway. GNN Replan-Each-Round + GNN Diagnoser
and Transformer Replan-Each-Round + GNN Diagnoser choose a new inspection after each
observation. The Transformer selects evidence and does not replace the shared final
GNN diagnoser. Full Recording + GNN Diagnoser uses a recording rule for evidence
selection and the shared learned model for diagnosis. Keep internal method keys
unchanged. Archived reports and sealed snapshots retain legacy display names.

Use all three B200s as independent training slots. Preserve the three legacy
experiment/result directories unchanged. Keep32 training,8 validation,8 seen,
16 unseen and8 larger unseen topologies; three seeds; all six core methods;
B∈{1,2,4} additional executions; the full registered case counts. Train with
100k maximum updates and validation-only patience10 at2000-update intervals.

Preserve compatible real source shards with an auditable import. Complete engineering
and leakage checks, then proceed through formal collection, training, frozen tests,
repair/audit and the four main figures. The optional appendix cannot delay main
results. Use actual measurements to forecast; inspect avoidable duplicate simulation
and I/O if the forecast becomes weeks. Never invent passed tests or finished jobs.
No favorable result for any method is required. Commands, logs, configs, hashes, failures and
actual progress must remain available in `/home/leiyo/work/cgx_data_v2`.
