# 04. Models, trainable quantities, rewards and optimization

## Three different objects

q_phi(y|H,G) predicts the diagnostic target. Q_psi(H,G,a,b,k) predicts the remaining
log-score improvement after choosing action a. pi_psi selects a feasible action using
Q_psi (or STOP). The policy is not a third separately trained neural model.
The graph encoder produces embeddings; the action head produces a scalar per candidate.
The same parameters handle different node/edge/action counts.

## Input representation

Encode each observed port's event sequence with a GRU (event embedding 32, hidden 64).
Deterministic evidence preprocessing must preserve matched-message comparisons using
only both endpoints when actually recorded. Use local timestamp differences, declared
units and available calibration readings; never compute a “corrected timestamp” with
hidden true offsets. Map opaque IDs to within-transcript equality/match indicators,
not numeric values. Include missingness explicitly; missing and numerical zero differ.

Per component include: type, port payload width/mask, nominal transform/limits, currently
acquired port summaries, sampling configuration, missingness, and observed matching
statistics. A declared clock group is represented relationally, not as an ordinal number.
Add virtual clock-domain nodes and membership edges if needed. They carry no hidden offset.
They are excluded from physical system size counts and cannot be fault locations.

GNN: width 128, three message-passing layers, edge width 32. For a directed dataflow
edge use a separate reverse relation for information propagation. Domain membership
edges have their own types. Edge attributes are public nominal role/shape/clock-relation
information. Use shared message/update MLPs, layer normalization and residual updates.
Use mean pooling plus log node count for a global summary. Keep endpoint embeddings for
local actions and valid-target scoring. Standard torch index_add/scatter_reduce is enough;
no requirement to install PyG or compiled custom CUDA extensions.

A trace action is represented by its selected ports/endpoints, rates, calibration domain
pair, public cost vector, current budget and remaining horizon. Pool its selected endpoint
embeddings; do not concatenate a fixed number of all graph nodes. Full-trace pools all
selected ports. Separate STOP returns exactly zero future improvement in the closed-loop
Q objective. Candidate feasibility is a deterministic external mask, not learned.

## Diagnostic predictor

Use a graph-conditioned reader as the common primary q. It scores each valid (component,
mechanism) with shared weights plus a global healthy head. Softmax over the legal target
list; do not use separate independent binary probabilities as if they summed to one.

Train with joint target cross-entropy divided by ln(2), averaged by case and family so
that long histories and large graphs do not accidentally dominate. Include initial-only,
partial, repeated, paired and full acquired histories. Validation records are independent;
no test history contributes to training, temperature selection or checkpoint selection.
Use FP32 log_softmax for scores/loss. If probability floor 1e-8 is needed, renormalize and
apply it everywhere (reward, validation and test), and record its use.

Cross-fitting by source topology (3 folds): reward and q-input features for a training
case are produced by a q not trained on that graph. All branches remain in the same fold.
Train a final deployment q on all source graphs after model choices are fixed. Use the
same final q for every policy seed and method. Record source-validation behavior under
OOF q and final q: a reader change can change what a policy chooses. This is not a new
source of test training. Use one canonical diagnostic seed for the core policy comparison
and label policy-seed uncertainty accordingly.

Train an independent relation-Transformer reader (width 256) on the SAME source evidence
access, with its own seed. At evaluation it re-scores each frozen acquired transcript.
It must not acquire new observations or change the policy that created the transcript.

## Closed-loop Q training

For a training transition (H_t,a_t,H_{t+1},Y), use

    r_t = log2 q_phi(Y|H_{t+1},G) - log2 q_phi(Y|H_t,G).

The label Y is used only by the trainer/evaluator. Do not feed Y, r_t, or actual loss
as an online policy input. q is frozen during Q training. Rewards can be negative; do
not clamp them to zero. With gamma=1 they telescope to final-minus-initial log score.
This is predictor improvement, not automatically mutual information.

The initial implementation is finite-horizon fitted Q-iteration with a target network:

    a_star = argmax over feasible a' of Q_psi(s',a',b',k-1)
    target = r + 1[not terminal] * Q_psi_minus(s',a_star,b',k-1)
    loss = mean Huber(Q_psi(s,a,b,k) - stop_gradient(target)).

Include G, acquired-history summary, budget vector and remaining decisions k in state.
Use target gamma=1, terminal bootstrap=0, STOP value=0. Updates are AdamW, lr=3e-4,
weight_decay=1e-4, batch=128, grad_clip=1, 200k maximum updates per policy run;
copy target weights every 1000 updates. Checkpoint/validate every 5000 updates.
Select the checkpoint with smallest source-validation final log-loss, pooled equally
across validation configurations; tie-break by lower actual resource use then earlier
checkpoint. Validate via full acquisition rollouts, not TD loss alone.

Behavior dataset is common across methods. Log action support by family/graph/budget;
reserve training-only branches for action-value prediction checks. Function approximation
can overestimate rarely seen actions. Record this failure, and implement adequate source
branch coverage rather than secretly searching test simulators. No claim of exact Q*.
Any conservative penalty or changed RL method is a versioned secondary experiment,
not an unreported replacement of the primary method.

Training resource/horizon sampling includes execution caps 1,2,4,8 (probabilities
0.1,0.1,0.7,0.1) and corresponding decision cap R+2. This enables the scheduled horizon
sensitivity without testing a completely untrained horizon encoding. Primary evaluation
is R=4. No new weights for each test budget or target graph.

## Generic learned comparator

Use a relation-aware Transformer, width128, three layers, four heads. Input includes
the same node evidence, nominal metadata, clock membership and connection information.
Use adjacency/edge-type attention bias or explicit edge tokens with consistent endpoint
references; no absolute node-ID embeddings. Use the SAME action and diagnostic heads,
update counts, replay cases, feasible masks and validation effort. Match parameter count
within about 20% where practical and report actual values/compute rather than hiding a
mismatch. Different attention connectivity is the intended architectural comparison.

## No-feedback learned plan comparator

`gnn_fixed_plan` reads G and X0 once and constructs an affordable plan before executing
any new observation. Its planning state includes remaining budget, horizon and chosen
planned actions, but no intermediate measurements, diagnostic updates, rewards or stop
signals derived from those measurements. When the plan is complete, execute it and
run the same final q on ALL collected records. Thus inference remains fully Bayesian-
style/updateable; only test selection is nonadaptive.

Train the plan value network with its own no-feedback transition representation and
terminal reward log2 q(Y|H_final)-log2 q(Y|X0); intermediate planning rewards are zero.
Use the same training cases/action library and source execution budget, and the same
200k update cap. Do not call it “optimal fixed”; it is a learned strong fixed-plan
comparator. In small finite fixtures exact fixed plans remain the separate reference.
A secondary open-loop replay of the adaptive policy's most likely branch is not a
substitute for training this comparator competently.

## One-step and two-step comparators

`gnn_myopic` fits expected immediate log-score improvement from the same transitions;
choose maximum feasible estimated improvement. Use GNN backbone and public structure:
this comparison isolates lookahead, not information access. Fit an additional next-loss
head on -log2 q(Y|H_next) for the cost-aware scoring adaptation.

`gnn_depth2` is the same Bellman scheme with lookahead capped at two decisions on each
choice and receding execution. Train horizon heads 1 and 2 from the same data; no hidden
test-case simulator is queried. Report it as a learned short-horizon baseline, not exact
model-based planning.

The 2024 cost-aware rule adaptation scores normalized predicted next-loss and declared
cost, beta in {0,.25,.5,.75,1}. Select beta on source validation. For a scalar resource cost
use each candidate's fractions of the remaining execution/byte/calibration caps, summed
with equal weights; disclose this choice. Its predicted next-loss is not true posterior
entropy, so do not claim literal reproduction of the 2024 model-based algorithm. In old
finite exact benchmarks it can use the known posterior as a separate correctly named run.

## Required model tests

Permutation equivariance for node/target/action scores; correct variable-size batching;
zero reward for no new observation; reward telescoping; terminal/STOP target; no hidden
label in features; consistent q across methods; exact-budget masking; finite non-NaN
softmax over all legal labels. See acceptance checklist for tolerances and full tests.
