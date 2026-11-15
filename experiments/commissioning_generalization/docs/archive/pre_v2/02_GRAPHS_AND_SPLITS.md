# 02. Graph generation and held-out systems

## What counts as a node

A physical graph node is an executed sensor/cache/adapter/controller/queue/actuator/
splitter/merger component. Counts 8–16 (source) or 24–32 (size test) exclude optional
representation-only observer-clock nodes. An individual component can expose two ports.
Observe ports; encode components and ports consistently, without pretending each record
is a new system component.

Dataflow has a state-side DAG from sensor to controller and command-side DAG from
controller to actuator. The controller/robot feedback loop is closed by the simulator;
the per-tick software schedule remains deterministic. Routes must preserve typed payloads.
Generate chain, one split/merge, and two disjoint split/merge motifs with 2 or 3 branches.
All motif kinds and component/edge types used in the primary test occur in training.
Branches partition the seven joint channels and execute real independent caches,
transforms or queues before merging. Do not invent arbitrary invalid graph edges.

Each graph has >=1 cache, >=1 queue and >=1 adapter. At least two applicable fault
locations per family are preferred across the source collection; never invent a
location class on a node that cannot execute that fault. Variation includes number,
position, state-vs-command adapter placement, and branch membership. Keep controller
and actuator identifiers semantic type roles, not persistent numeric positions.

## Generator algorithm

1. Seed a candidate stream deterministically from the manifest seed and requested suite.
2. Pick component counts and a valid source motif; assemble typed ports and edges.
3. Insert cache nodes only on state paths, queues only on command paths, adapters on
   either side. Assign consistent invertible representation conversions to interfaces.
4. Assign 2–5 observer clock domains (size suite 3–8) with at least one within-domain
   and one cross-domain timing pair. Domain names are arbitrary local symbols.
5. Validate DAG schedule, type compatibility and channel coverage. Run the same healthy
   qualification protocol, record failures rather than silently hide rejection counts.
6. Check graph identity against already accepted source/held-out manifests. Node renaming
   and arbitrary clock renaming do not create a new structure.
7. Accept in deterministic order until required counts are reached; if the generator
   cannot fill a suite, report the actual obstruction, do not duplicate/rename graphs.

Use a labeled graph hash as a candidate filter, followed by exact labeled isomorphism
for collisions (e.g. NetworkX GraphMatcher). A Weisfeiler–Lehman hash alone is not a proof
of non-isomorphism. Primary structure identity includes component type and port/dataflow
roles, but NOT random IDs, sampled continuous parameters, faults, or clock names.
Clock-partition identity is a separate isomorphism-invariant equivalence relation.
Save canonical manifests and exact split assertions.

## Counts and matching

Authoritative counts are in protocol.json: 128 train, 24 validation, five 20-config tests.
Training cases: 256/config, with 64 per family. Validation/test cases: 128/config, with
32 per family. For each nonhealthy family, select legal node/severity through deterministic
balanced shuffling (counts within a type differ by at most one where possible).

Select 20 training structures by a deterministic pre-training rule as anchors:
- test_seen: same structures, new cases/tasks/offsets/noise; no case overlap.
- test_connections: each graph matched to one anchor in node count, type counts and
  branch count; rewire/reorder valid components to a graph not isomorphic to ANY training
  or validation graph. Match degree histogram when feasible; report deviations explicitly.
  Hold fault/noise/clock-domain-count distributions within source ranges. This is the
  PRIMARY transfer test; do not also introduce new component types or larger sizes.
- test_clocks: reuse anchor dataflow but use a clock partition withheld for that graph.
  Keep domain count, domain-size distribution and cross-domain-edge count matched where
  possible. True clock offsets are still hidden. This tests a new measurement relationship,
  not entirely new dataflow topology.
- test_size: 24–32 components, familiar types/motifs, new graphs; report absolute and
  normalized resource costs. It is a scale shift separate from the primary comparison.
- test_implementation: anchor topology/clock partition, independently coded B mechanism,
  new cases. This is implementation transfer, not topology novelty.

Clock/implementation tests intentionally share an anchor graph. A validator that requires
all graph hashes globally disjoint is WRONG. Require the appropriate composite identity
for each suite and no overlapping case/task/random stream. Seen is not mislabeled unseen.

## Public input versus private manifest

Public: component and edge types, nominal port dimensions/units/transforms, clock-domain
membership, allowed sample rates, action costs, requirements, nominal task schedule.
Private: fault family/location/severity, actual offsets, injection code path, original
case/configuration identifiers used by the generator, oracle trajectories and labels.
Evaluation bookkeeping IDs may appear outside network inputs for joins and grouping.

Randomly relabel component indices in training and at evaluation setup; remap edges,
labels, ports and action endpoints consistently. Do not supply topology_id embeddings,
absolute node-position embeddings, raw numerical clock IDs or numeric message IDs.

Use separate seeded streams for graph generation, case selection, tasks, disturbances,
clock offsets, timestamp noise, behavior actions, model training and hidden audit.
Use a stable hash/SeedSequence, never Python's process-randomized built-in hash().
Counterfactual training branches inherit their parent's split and case-group identity.

## Variable target space

For graph G, construct legal targets from public component type:
  {healthy} union {(v,cache_stale)} for caches union {(v,queue_hold)} for queues union
  {(v,adapter_scale)} for adapters.
The index list can change with G. Scoring uses a shared per-target head and a valid-label
mask. Normalize softmax jointly across this list. A topology change must not require
resizing a classifier's final Linear layer. No unseen mechanism labels in the main test.

The prior gives healthy 0.25 and each mechanism family 0.25, then distributes uniformly
over its legal locations. It is known to all methods. Compute its log-loss baseline per
G. Larger graphs have more target entropy: report residual prediction loss and relative
to this baseline, not just raw cross-graph log-score improvement.

## Sealing

Save split_manifest.jsonl, public_graphs.jsonl, private_cases.jsonl, tasks.jsonl and hash
manifest before policy training. Private evaluator files live in a different artifact
subtree. Public policy calls receive immutable whitelisted views, not a whole simulator
object containing a `.truth` property. Save protocol_lock.json with all hashes and a
record of test access. No graph selection after inspecting model performance.
