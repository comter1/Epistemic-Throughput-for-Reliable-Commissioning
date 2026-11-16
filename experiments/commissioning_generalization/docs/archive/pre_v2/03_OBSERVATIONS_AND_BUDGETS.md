# 03. Observation actions, budgets and stopping

## Primitive evidence

Every module input/output port can record a fixed-schema event: local timestamp, opaque
message token (a composite token for merged channel groups), event type, subset mask, observed
numeric payload, and missing/timeout flags. Node/type/port metadata are public; injection
parameters are not. A legal observation cannot read complete hidden logs afterwards.

Tracing selects rates 10,50,100 Hz over a 10 s execution. Use an internal scheduling grid
to decide which events to log, but never output the global tick index. Cache repeats
reuse the original opaque message identity; transformations preserve provenance.
After split/merge, derive a fresh opaque composite token from the internal constituent
IDs and channel masks; downstream ordinary cache/adapter/queue operations preserve it.
Do not assume a merger input and its output share an identical token. Internal provenance
tuples are not automatically available to the agent. If an endpoint was not recorded,
the agent cannot access its matching event. Numeric
content may also carry information legitimately; never wipe it merely to force calibration.

## Finite action catalog generated from G (not from truth)

- single_port(v,p,f): record one port at f.
- module_pair(v,f): both ends of one executable cache/adapter/queue at f.
- link_pair(u,p,v,q,f): endpoints of a public dataflow link at f.
- pair_plus_calibration(pair,f,d_i,d_j): one of the same legal pairs plus a burst for
  its two different clock domains. If domains already match, this is not offered.
- calibration_only(d_i,d_j): burst for a distinct-domain pair appearing in a legal
  timing comparison; no robot execution, one decision and one calibration.
- full_trace(f): all public trace ports at a common f, no privileged channel. It is
  offered to every method and subject to the same byte/execution caps.
- STOP: zero extra cost, ends acquisition.

For a local pair, at most four ports are permitted (e.g. two-module boundary grouping).
Do not enumerate arbitrary powersets. Canonicalize duplicate action content; a port
record requested twice in one action is recorded and charged once. One action has at
most one calibration burst. Number of actions varies by graph; no fixed output dimension.
A full-trace reference may separately choose calibrations from the same catalog.

## What changes after an action

A trace-producing action configures instrumentation BEFORE a fresh 10 s execution.
Receive selected port records plus a costed default external outcome stream; update
history. A calibration-only action returns four-timestamp exchange records without
resetting/rerunning the robot. Combined actions return both, using the same persistent
case offsets but independent measurement errors. New reads do not change faults or
physical clock offsets. A later execution is reset to the same task initial state with
fresh indexed disturbances; it does not continue from a diagnostic-run end state.

## Wire format and byte budgets

Use an actual fixed-size 128-byte binary event record, not a made-up scalar “cost”.
A concrete v1 little-endian format is `struct.Struct("<d16sIBBH14f40x")` (128 bytes):
local timestamp(float64), opaque token(16 bytes), local port index(uint32), event code
(uint8), flags(uint8), joint mask(uint16), payload(14 float32 values), fixed padding.
Unused payload entries are zero with a known port-width mask, not treated as observations.
A port index is a local lookup reference, never a learned persistent embedding.
Implement this format or an exactly documented equivalent and test its actual length:
local time, opaque provenance, port reference, flags and masked payload fit the record.
A concrete burst format `struct.Struct("<16d4Q")` is 160 bytes.
A 160-byte calibration burst contains 16 float64 timestamps (128 B) plus 32 B of
public pair/flags metadata. Emit real serialized bytes and count them. File compression
is additional measured storage cost; budget is uncompressed transmitted wire bytes.
Do not charge decoded embedding size instead of acquired raw data.

Let P_G be number of trace ports. Let R_ext be bytes from the mandatory external stream
per execution. A nominal full-rate execution has
  W_full(G) = 1000*128*P_G + R_ext.
Calibrations are separate, not assumed present in full trace.
Set C_ref(G)=4*W_full(G) and byte cap=floor(f*C_ref(G)), f in {0,.1,.25,.5,1}.
This denominator is frozen before observing the case and remains 4 executions even
when the execution-cap sensitivity uses 1,2,8. Report absolute bytes as well as f.
The calibration burst's 160 bytes also consume this byte pool, in addition to calibration
count. Base X0 acquisition is common, outside incremental cap but included in total cost.

A candidate action has a public conservative byte bound from format, duration, selected
rates and ports; mask it if bound exceeds remaining cap. Truncation caused by unanticipated
extra events is visible and charged; do not retrieve omitted tail records later.
A fixed-rate snapshot logger is preferred to an unbounded event logger for primary runs.
Unused budget is permitted. Actual bytes can be below bounds; score against actual use too.

## Main caps and sensitivity

Main: extra executions<=4, calibration bursts<=2, decisions<=6, wire bytes<=cap.
A trace action uses one execution and one decision. A calibration-only action uses no
execution and one decision. A combined action uses one of each, still one decision.
STOP is not charged. Every cap must hold on every trajectory.
Sensitivity: executions 1,2,8; decisions=executions+2; calibration cap=2 and SAME byte
reference denominator. This prevents “more rounds” quietly giving more recording budget.
Use the same predeclared evaluation subset for runtime-heavy sensitivities, not cases
selected because adaptivity performed well.

The main learning objective is not IG divided by cost. Cost constraints are explicit.
For a cost-sensitive baseline use a separately documented score/weight, selected on
validation. Do not silently replace the common feasible action set.

## Runtime limits

Selection and diagnostic forward calls each have a 2 s wall-time cap on the reported
hardware, with clean warmup excluded and reported. On timeout, return last normalized
prediction (or fixed prior), STOP, operationally abstain and retain the case in metrics.
Do not drop stalls. Runtime comparisons use the same CPU/GPU allocations and batching.
Batch evaluation timing and latency for one live decision are distinct reported quantities.

## Full-recording controls

`full_recording_feasible` buys a predeclared full_trace plan only if it fits every common
cap; lower rates can be chosen at the start using public costs and initial evidence.
No per-test truth-based choice. `full_recording_unrestricted_reference` uses four full-rate
runs and the allowed calibration count even when the byte cap is exceeded. Label it
infeasible/reference at those caps, never plot it as a feasible policy on the same frontier.

## Encoding does not change access rights

Raw evidence may be compressed to <=256 tokens per port/run by deterministic binning:
include counts, first/last, mean/min/max, repeat-ID statistics and matched residuals.
The algorithm is fixed before testing; it may use only acquired fields. Measure raw cost
before compression. Hidden evaluator logs cannot supply missing bins or correct timestamps.
Same encoder/tokenization budget for every acquisition method using the shared reader.
