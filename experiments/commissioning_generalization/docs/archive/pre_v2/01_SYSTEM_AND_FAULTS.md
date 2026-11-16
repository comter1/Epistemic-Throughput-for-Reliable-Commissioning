# 01. Executable system, tasks, faults and clocks

## Physics and asset

Use native CPU MuJoCo Python bindings; no renderer/viewer in the main path. Load Panda
from a pinned Menagerie commit, retain its license and asset hash. Set physics dt=0.002 s.
Run the experiment synchronously, as fast as the CPU allows; never use time.sleep to
simulate queue delays. Record actual host wall time separately from simulated time.

Each 10-second execution starts from a fixed safe nominal joint pose and zeroed controller
and pipeline transient state. The case's fault and observer offsets persist across runs.
Fresh disturbances use an execution-indexed random stream shared across compared methods.

## Task and a controller that actually uses the potentially faulty state

Generate three reachable waypoint joint configurations around the standard Panda home
pose. One implementation can use deterministic damped-least-squares inverse kinematics
on a separate clean kinematic model to create the catalog; save all references before
runs. Never solve IK using an unacquired diagnosis signal. No policy chooses the task.
Use the same finite task catalog/distribution for every topology and method, with separate
source/test/audit random seeds. Reference end-effector locations are nominal FK of these
configurations. A fixed smooth reference q_ref(t) moves between waypoints and then holds.

At 100 Hz, the high-level controller receives only the state packet delivered through
the configured sensor/cache/adapter graph. It computes, in canonical radians,

    u_t = clip(q_hat_t + 0.20*(q_ref(t)-q_hat_t) - 0.02*qdot_hat_t,
               joint_lower+margin, joint_upper-margin).

Route u_t through the command adapters/queues before applying it to the Panda position
actuators. Finger commands remain at a fixed safe setting. Preserve declared actuator
limits. No learned control. This explicit state-feedback term is required: directly
sending a precomputed q_ref to the actuator would make a stale input cache irrelevant.
The low-level physics actuator's own PD feedback is part of the fixed plant, not leaked
to the diagnosis policy.

Choose waypoint timing so each target has a >=0.5 s hold interval before its deadline.
Suggested schedule: settle [0,0.5], move/hold to targets by 3.2, 6.4, 9.6 s; end at 10 s.
The generator must produce healthy executions reaching <=0.02 m error for 0.5 s, with
joint-limit tolerance 0.001 rad, no prohibited contacts and valid command limits.
Verify the frozen controller on source-only healthy qualification fixtures. If a global
bug/gain fix is necessary, document it BEFORE sealing manifests; no per-test tuning.
Healthy qualification of generated graphs uses an identical preset criterion. Keep
rejection reasons/counts. Never reject a fault case because it is hard to identify.

## Real modules and typed ports

Every component has typed input/output ports, a step(packet,sim_time) operation, and
an observation hook. Nominal parameters/specifications may be public. Injected parameters
are private and are never serialized into agent packets.

- sensor: reads q and qdot with the declared sensor noise; emits an opaque message ID.
- cache: stores/forwards the most recent available packet under nominal behavior.
- adapter: applies a declared invertible unit/affine representation transform.
- controller: consumes canonical state and emits canonical commands.
- queue: schedules/delivers packets, nominally without an extra control-tick delay.
- actuator: receives canonical joint targets and sets MuJoCo controls.
- splitter/merger: partitions/recombines disjoint joint subsets, not copies for show.
  A merger uses latest available per-joint values and preserves origin IDs per subset internally. Public logs use one opaque composite token
  after a split/merge; they do not expose a variable-size tuple without charging it.

Packets have simulation-private scheduling metadata, and a separate public-log projection.
Never give private scheduling time/ticks to the agent. Boundary recorders see local observer
time, payload and opaque matchable IDs only. All clock domains remain physically fixed;
clocks belong to recorders, not robot controller timing.

A nominal conversion from representation (a_in,b_in) to (a_out,b_out) applies
x_out=a_out*((x_in-b_in)/a_in)+b_out componentwise. Use consistent nominal contracts at
all edges. Merely moving an adapter without repairing the representation contract is
not a legitimate new topology. Split/merge channel masks cover each required joint once.

## Faults: one per case, location is part of the answer

Families are healthy, cache_stale, queue_hold, adapter_scale, with equal family mass.
Select a legal node uniformly within the chosen family; balance positions across cases.
Within a case select severity and keep it fixed across all diagnostic executions.

cache_stale:
  Implementation A uses a ring buffer and returns the state from 2,5,or10 control ticks
  earlier. At startup, use the earliest available packet and log the actual repeated ID.
  Fault changes controller input, not only timestamps. Repair restores latest-packet
  consumption and clears stale internal state through the ordinary reset interface.

queue_hold:
  Implementation A schedules each command at send_time + delay, where delay is 2,5,or10
  control ticks. Deliver available commands in timestamp order with a declared latest-
  command policy; hold the previous valid command when none is due. Do not simulate an
  unbounded backlog. Repair restores nominal release and expired-message handling.

adapter_scale:
  A single adapter applies an extra gain 1 + sign*magnitude, with magnitude 0.05,0.10,0.15.
  The public specification remains nominal. Apply to the actual transformed numeric
  payload, so behavior changes downstream. Repair restores the declared transform.

Implementation B is independent code with the same mechanism: scheduled snapshot reads
instead of ring-buffer indexing; bucketed delivery schedule instead of a heap; matrix
application instead of elementwise scaling. Its semantics must pass independent unit
reference tests; implementation identity is not an agent feature. No source training
on B. Independent code path does not mean a different allowed fault family.

Some faults may have small task impact in a particular trajectory. Include them, report
impact stratification, and never filter them after seeing diagnosis results. One can
have a real malfunction without catastrophic task failure. Insertion of dummy nodes
whose behavior never affects a task is prohibited.

## Observer clocks and calibration

At an event with private simulation time t, public timestamp from domain d is

    t_obs = t + delta_d + epsilon_event,
    delta_d ~ Uniform(-20,20) ms, epsilon_event ~ Normal(0,1^2) ms.

Offsets persist within the case. Do not make every recorder's first sample occur at a
publicly known identical simulator time: that would provide a free synchronization marker.
Each domain's recorder arms after an independently sampled 0–100 ms setup delay and a
sampling phase uniform within one sample period, freshly per execution and keyed by
case/run/domain. These setup times are not returned. Charge/record all actually acquired
bytes; bound counts conservatively by full-duration rate. Matching across acquired opaque
IDs is allowed; global start/end barriers, simulation ticks and per-port true-time sample
indices are not. Nominal task timing is public, but actual logger schedule is not.
An agent may legitimately infer timing from acquired signal patterns; do not forbid that.
Event noise is independent across recorded endpoints
in the main condition. The shared-round exact-cancellation setting remains only in the
old toy; it is NOT the main-system noise mechanism.

Calibration estimates a difference between two clocks using a four-timestamp exchange.
For domain i sending to j:
  t1 = clock_i(send); t2 = clock_j(receive);
  t3 = clock_j(reply); t4 = clock_i(receive reply).
  delta_hat_(j-i) = ((t2-t1)+(t3-t4))/2.

One calibration action performs four exchanges. Forward and backward transport delays
are independent positive lognormal variables, mean 2 ms, SD 0.5 ms. Use
sigma_log^2=log(1+(sd/mean)^2), mu_log=log(mean)-sigma_log^2/2.
Receiver reply processing delay is 0.1 ms. Each timestamp has its independent recorder
noise. Raw t1..t4 values and declared endpoint domains are returned, not true delta.
The residual includes half the forward/backward asymmetry; it does not cancel perfectly.

Calibration neither synchronizes physical clocks nor modifies controller/pipeline state.
Its measured CPU wall time, simulated exchange duration, wire bytes and burst count are
reported. Do not create fake sleeps or an arbitrary financial penalty. Calibration
messages use the observer link, not the potentially faulty command queue.

## Default and evaluator-only instruments

The common initial execution provides a deployable external end-effector trace at 10 Hz,
reference trajectory and requirement flags, with the same measurement noise contract.
Do not expose internal joint truth as a free global diagnostic input. Subsequent trace-
producing runs have a costed external record stream too. The hidden evaluator may inspect
full physics truth to score task requirements, but it never supplies extra fields to
selectors, predictors, or the public verifier.

Repairs are ordinary configuration calls addressed to the predicted node. They do not
look up the injected fault or simply clear an evaluator fault flag. A wrong-node repair
must leave the actual fault unchanged. Reset-all is a separate explicitly costed baseline.
