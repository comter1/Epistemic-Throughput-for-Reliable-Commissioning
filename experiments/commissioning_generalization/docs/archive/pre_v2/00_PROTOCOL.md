# 00. Frozen main-experiment protocol

## Main question

Train on source configurations of an executable robot software pipeline. Freeze all
weights and inference preprocessing. On held-out configurations, choose paid observations
and clock-offset measurements using only public structure and acquired records. Test
whether this obtains useful diagnostic evidence at lower resource use than strong rules,
planning approximations and a generic learned policy.

The central claim is **generalization of learned acquisition decisions to unseen
connections**, not invention of active diagnosis, RL, GNNs, or mutual information.
We test known component types assembled in new valid ways. We do not claim arbitrary
transfer from a robot to a printer, or diagnose entirely new fault types in the primary study.

## What is held fixed and what changes

Fixed: Panda physics, task distribution, low-level actuator specification, controller
code/gains, component type vocabulary, fault families, input feature schema, acquisition
rules, target meaning, cost accounting, evaluation requirements.

Changes across configurations: component multiplicities and positions, legal dataflow,
split/merge placement, observer-clock membership, and (in separate suites) scale or
implementation. Changes within a case: freshly sampled disturbances and measurement
noise; hidden fault, severity and clock offsets persist until explicit repair.

## Core objects

G is the public typed interface graph. Theta is hidden case state (fault location and
mechanism, severity, actual observer offsets). Y is `healthy` or a valid `(node, mechanism)`.
X0 contains G, nominal specifications, requirements, costs, task schedule and one common
initial execution observed through the default external interface. The agent has history
H_t=(X0,a_0,o_1,...,a_{t-1},o_t). A decision can collect calibration evidence without a
robot rerun, or configure ports and perform a fresh robot execution, optionally with one
calibration burst. STOP is always legal.

We distinguish a **decision step**, a **robot execution**, and a **calibration burst**.
Main caps: 6 decisions, 4 extra robot executions and 2 calibration bursts. This replaces
ambiguous uses of “round”. At most four trace-producing reruns, not six, are permitted.
All resources are pathwise caps. No averaging of overruns across cases.

## Scientific quantities

The information reference is G_pi(b)=I(Y;H_tau | X0), measured in bits within a budget.
In the finite toy, exact posteriors/DP allow genuine MI and restricted exact optima.
In MuJoCo, the main observable is
L_pi(q)=E[-log2 q(Y|H_tau,G)]. The relationship is
L_pi(q)=H(Y|X0)-G_pi+E KL(p(Y|H_tau,G)||q(Y|H_tau,G)).

Thus a change in learned prediction score is **not automatically an information gain**.
Use “log-score improvement” on main-system plots, not “exact ET” or “Bayes optimum”.
Independently re-read acquired transcripts with a different diagnostic model to test
whether improvements depend on one reader. This is a robustness check, not proof of MI.
Do not claim exact optimum in a large simulator from beam search or a learned value model.

## Fixed contribution and hierarchy of evidence

Primary: unseen-connections suite; final log-loss at byte fraction 0.25, execution cap 4,
calibration cap 2. The primary paired comparison is gnn_adaptive versus transformer_adaptive.
Strong rules and planning controls remain essential comparisons. Superiority on this
comparison is not guaranteed and is not required for honest completion.

Also report every budget point; seen-to-unseen performance change; mechanism+location
accuracy; actual resource use; per-configuration variability; the fixed-plan feedback
ablation; independent-reader results; and downstream repair/audit outcomes.

Primary inference uses systems, not individual video frames, as independent units.
GNN superiority, adaptivity value and cross-system utility are separate conclusions.
A weak GNN-vs-Transformer result does not imply acquisition itself has no value. A good
same-system score does not establish transfer. A neural network accepting a bigger
matrix does not establish it can make useful decisions on the bigger system.

## Source of truth and changes

`configs/protocol.json` is the numerical authority. Definitions in these documents are
normative. Freeze graph/case manifests, schema, software/asset versions, and configuration
hashes before training. Validation selects checkpoints and the explicitly listed rule
weight/acceptance threshold; it cannot select the held-out system collection.

Any change to target, action library, faults, noise, budget units, split, or main metric
creates a versioned amendment before looking at test outcomes. Rerun affected comparisons.
Hardware-only changes (worker count, safe batch-size microbatching) may preserve protocol
if samples/updates remain equal and all timing is relabeled.

## Engineering qualification is not an exploratory study

Small deterministic checks must establish healthy behavior, executable faults, proper
repairs, message identity, observation isolation and resource accounting. These checks
are pass/fail software tests. They do not choose favorable hypotheses or permit replacing
main experiments with another toy. Once technically valid, continue to formal training
and testing regardless of whether a preliminary method beats another.
