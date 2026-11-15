# MAIN EXPERIMENT V2 implementation

The active protocol is CGX-2.0; the old implementation report is archived in
`docs/archive/pre_v2/IMPLEMENTATION_REPORT.md`. `STATUS.md` and external artifacts
are the authority for actual execution, not this implementation inventory.

The implementation retains real headless Panda MuJoCo dynamics, typed software
routing, physical cache/queue/scale faults, ordinary repairs, costed records,
noisy clock measurements, topology cross-fitting, public-only inputs, variable
node/action/target heads and the common frozen GNN diagnoser. The inspection
selector in Transformer Replan-Each-Round + GNN Diagnoser receives the same public
relations as the GNN selectors; its final diagnosis still uses the shared GNN.
No test-time fitting is implemented. [Method descriptions](docs/METHODS.md) separate
each evidence selector from this common final diagnoser.

V2 adds deterministic outcome-independent projection of32/8 source topologies,
all3200 registered test cases, strict provenance-checked old-shard imports,
three-seed15-job policy manifests, validation-only early stopping, execution-budget
conditions B=1/2/4, and the six-method main matrix. The source-only support branch
for GNN Plan-Once + GNN Diagnoser retains its original0.25 byte fraction; this is independent of
V2's execution-budget evaluation axis. All generated artifacts stay outside Git.

Runtime changes include independent CPU validation/evaluation/repair workers,
lossless memory-mapped features, batched OOF scoring, and a detached SSH worker
for the third B200. Jobs preserve exit codes, heartbeat, GPU memory, update counts,
checkpoint hashes and actual interval throughput. Resume verifies artifact identity;
an early-stop marker requires the registered sequence of actual validation checks.

Frozen acquisition results feed repair/audit without reacquiring diagnostic evidence.
Independent readers consume the exact same transcript hashes. Paired statistics,
seen/unseen gaps, resources to a validation-fixed quality target, and repair/audit
costs generate four prioritized main figures. Optional appendix stages do not block
main completion. Negative outcomes and timeout denominators are retained.

Engineering smoke artifacts are isolated and never scientific results. Older
qualification and failed/replanned runs remain in external logs and archived reports.
Current measurements and forecast assumptions are saved before formal launch.

Public reports use the full method names in `docs/METHODS.md`. GNN Plan-Once creates
a learned inspection plan for each new system and executes it without intermediate
replanning. GNN Replan-Each-Round and Transformer Replan-Each-Round choose a new
inspection after every observation. GNN 2-Step Planner considers one additional
future inspection, and GNN Cost-Aware One-Step combines a learned next-inspection
estimate with declared cost. Full Recording selects evidence without a learned
selector. All six use the same final GNN diagnoser. Historical generated reports
and sealed snapshots retain legacy display names; numerical results and protocol
identifiers are preserved.
