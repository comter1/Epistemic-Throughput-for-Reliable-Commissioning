# MAIN EXPERIMENT V2 — CGX-2.0

This is the formal main experiment. The question is whether an evidence-acquisition
policy trained on source system configurations transfers, with frozen weights,
to structurally unseen system topologies. No favorable result for any method is required.
CGX-1.0 and the temporary single-seed P48 delivery proposal are superseded.
Their specifications remain under `archive/pre_v2` and are not executable defaults.
`../configs/protocol.json` is the active numerical authority. The user's latest
hardware instruction supersedes the two-GPU wording in the pasted requirements:
three B200s are available (two local, one at `leiyo@ssh.cloud.sdu.dk:2127`).

| Split | Topologies | Cases/topology | Cases | Physical components |
|---|---:|---:|---:|---|
| Train | 32 | 128 | 4096 | 8–16 |
| Validation | 8 | 64 | 512 | 8–16 |
| Seen-topology test | 8 | 100 | 800 | 8–16 |
| Unseen-topology test (primary) | 16 | 100 | 1600 | 8–16 |
| Larger unseen-topology test | 8 | 100 | 800 | 16–24 |

Topology identity excludes numeric node labels, random seeds and continuous
parameter values. Train/validation/primary/larger sets are checked for labeled
structural isomorphism. Seen graphs reuse source structures but never source
cases/RNG streams. Primary graphs have familiar component/edge types, unchanged
fault/severity/noise/task distributions and nominal parameters. Two nonisomorphic
type-order/connection variants are matched to each of eight source anchors, with
component multiplicities/branch counts/degree distributions and clock membership
held comparable. Size shift is a separate test. Clock/implementation shifts are
optional appendix work and cannot block the main result.

Fault targets are healthy or (component, mechanism), with one physical injected
fault: stale cache, delayed queue or adapter scale mismatch. Every topology's
cases are one-quarter each family, with balanced eligible locations. The actual
Panda MuJoCo pipeline, fixed controller, sensing and observation formats are
unchanged. See the archived physical/interface specifications and the preserved
`PHYSICAL_IMPLEMENTATION_DECISIONS.md` for implementation details. No synthetic
diagnostic score replaces a physical execution.

The user explicitly confirmed that B means maximum additional diagnostic
executions: B∈{1,2,4}. Calibration is separately limited to two operations;
decision cap is B+2. A fixed safety ceiling of four full-rate recordings is kept
for bytes; actual bytes are reported separately. Main maximum is four additional
executions. STOP remains legal. Full Recording + GNN Diagnoser is one reference condition at B=4
with complete recording and its actual cost, not a duplicated baseline at every B.
Execution count, bytes, calibrations, selection CPU time and total acquisition time
remain separate quantities. Per-call deadlines and all failed/abstained cases are
retained. Optional horizon sensitivity at1,2,8 does not block main outputs.

The six deployed methods have the following public names. The first five use
seeds11,23,37. Their internal keys remain unchanged.

| Public method name | How the next inspection is chosen |
|---|---|
| GNN Plan-Once + GNN Diagnoser | A trained GNN uses the new system's structure, initial evidence and budget to create a complete system-specific plan before collecting new evidence. Later steps are not revised from intermediate observations. |
| GNN Cost-Aware One-Step + GNN Diagnoser | A trained GNN estimates the diagnostic value of each available next inspection; the selection rule also accounts for its declared cost. |
| GNN 2-Step Planner + GNN Diagnoser | A trained GNN chooses the next inspection while considering the value of one additional inspection afterward. |
| Transformer Replan-Each-Round + GNN Diagnoser | A Transformer reads the updated evidence and chooses the next inspection again after each observation. |
| GNN Replan-Each-Round + GNN Diagnoser | A GNN reads the updated evidence and chooses the next inspection again after each observation. |
| Full Recording + GNN Diagnoser | A recording rule collects the full allowed signals in the reference condition, without a learned evidence selector. |

The internal `gnn_myopic` network supplies GNN Cost-Aware One-Step + GNN Diagnoser;
its uncosted variant is not another main method. Its checkpoint validation uses the
cost-aware selection rule at beta0.5; beta is subsequently chosen from the fixed
validation grid. GNN Plan-Once generates a learned plan for each new system; it is
not one hand-written checklist used on every system. GNN 2-Step Planner's learned
value model is conditioned on one or two remaining decisions: "2-Step" describes
the planning horizon, not the number of neural-network layers. The Transformer
inspection selector gets the same public graph relations, evidence and feasible
actions as the GNN inspection selectors. Small model widths stay unchanged.

After evidence collection, all main methods give their collected records to the
same final GNN diagnoser with the same frozen weights. Thus the main comparison
changes how evidence is selected and collected, not the final diagnostic model.
A GRU reads each recorded interface sequence; a 3-layer GNN combines these
summaries along the system connections and outputs probabilities for healthy
operation and each legal (component, fault type) target. See [METHODS.md](METHODS.md)
for the actual model widths, full method explanations and internal-key mapping.
Three topology-cross-fit readers provide source-only rewards and q-inputs. A
separate independent Relation-Transformer reader re-scores the same frozen
transcripts; it is neither the main diagnoser nor the Transformer inspection
selector. Runtime never receives fault labels,
injection values, true clock offsets, unacquired records or evaluator state.
Reader/policy/normalization parameters and selection/repair thresholds are frozen
before opening held-out outcome files. No test-time fitting or gradient updates.

Training starts from approximately50k–200k transitions, using eight designated
behavior trajectories per base case plus only preregistered source coverage/support
branches. No million-transition minimum remains. The source execution cap is200k,
including compatible imported prior executions and interrupted fresh attempts.
Replay, reward telescoping, finite-horizon undiscounted DoubleQ/Huber, batch128,
AdamW settings, clipping and target copies are unchanged.

Every reader/policy has a100k update maximum. Validate and checkpoint every2000
updates. Stop after10 consecutive checks without at least0.005 bits of improvement
relative to the last meaningful improvement. This numerical delta is fixed before
training/test access. Save/select the actual lowest validation NLL checkpoint,
then lower measured bytes, then earlier iteration. Reader validation uses all
recorded validation histories with equal case/configuration weighting; policy
validation executes all512 cases at B=4. GNN Replan-Each-Round and Transformer
Replan-Each-Round use the same maximum,
patience, batch and validation rules; actual update counts/compute are reported.

The main matrix is147,200 case/method/seed/budget cells:
3200×(5 methods×3 seeds×3 B values +1 Full Recording reference).
The primary contrast is GNN Replan-Each-Round + GNN Diagnoser versus Transformer
Replan-Each-Round + GNN Diagnoser on the1600 unseen-topology cases at B=4,
using paired2000-replicate configuration-then-case bootstrap. Secondary
comparisons receive Holm correction. Training-seed variation is separate.
Report log loss, joint location/mechanism accuracy, abstentions, resources and
seen-to-unseen degradation. Learned log-score improvements are not exact MI.

Repair/audit uses a sealed, family-balanced subset of four configurations per test
suite and32 cases per selected configuration (384 cases). All six core methods
at B=4 are included, with three seeds for learned methods:6144 repair conditions.
Confidence thresholds are selected only on validation. Two fresh public verification
runs are followed, after declared completion, by20 independent hidden audit runs.
Reuse the frozen diagnosis/acquisition record; do not repeat diagnostic acquisition
just to perform a repair. Failed verification/audit outcomes remain in tables.

The four prioritized figures are: seen/unseen budget-performance curves; actual
resources reaching a validation-fixed quality target; seen-to-unseen gap; and
repair/audit outcome versus total deployed acquisition/verification cost. Unreached
quality targets stay marked unreached, with no test-derived target or extrapolation.

Original completed source shards are imported only when graph, task, case labels,
all RNG streams, observation and behavior-collection contracts match exactly.
The first32 train and first8 validation topology records are selected independently
of outcomes. Within each family, manifest-hash ordering and round-robin location
balancing choose the smaller case set. Import records include original marker,
transition, source protocol and new protocol hashes. Unselected old data is retained
outside the V2 training index. New test cases and the new split are sealed before
training. Generated artifacts remain outside Git; legacy experiments stay unchanged.

Runtime uses independent per-GPU jobs across all three cards, CPU process pools for
validation/test/repair, batched OOF inference, and a lossless memory-mapped feature
cache. No data precision reduction or sample pruning accompanies these changes.
CPU timing is labeled as CPU timing. Remote jobs run detached and retain telemetry,
exit codes and hashes in shared storage. Qualification, real throughput measurements,
commands, logs, job states and forecasts precede launch. Estimates are planning
numbers; no unfinished stage is labeled passed.
