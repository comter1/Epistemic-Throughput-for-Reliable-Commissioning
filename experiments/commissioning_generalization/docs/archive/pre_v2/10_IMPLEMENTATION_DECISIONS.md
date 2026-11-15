# Pre-lock implementation decisions

These decisions fix implementation details before the formal protocol and manifests
are sealed. They do not use held-out outcomes and do not change CGX-1.0's numerical
configuration, task, target, faults, actions, resource caps, split counts or main metric.

## Feedback-independent replay for the fixed-plan comparator

An action selected by an adaptive behavior rule can depend on evidence that a fixed
planner is forbidden to receive. Treating its realized action path as an ordinary
open-loop training plan introduces observational selection bias. Consequently,
`gnn_fixed_plan` uses only `random_legal`, `fixed_diverse_plan`, and the independently
scheduled training coverage branches from the common source corpus. All other methods
receive these same records too. Its inputs still contain only the initial observation,
public graph, planned actions and remaining conservative budget. Its terminal reward
is the common out-of-fold reader's complete plan log-score improvement, assigned to
the last real acquisition; STOP remains exactly zero.

Sampling first selects a source case uniformly, then an eligible non-STOP transition
within that case. Only prefixes affordable under the sum of public upper cost bounds
are eligible. Every source case must have support. If all eight designated behavior
trajectories of a case were adaptive rules, one extra independently chosen diverse
fixed-plan trajectory is included in the predeclared shared training-only coverage
pass. It counts toward the unchanged two-million-execution cap and is available to
all learners. The eight designated trajectories, behavior mixture, five policy seeds,
200,000-update cap, shared deployment reader and validation effort remain unchanged.
The eligibility and per-case counts are saved in `training/fixed_plan_support.json`.
A missing case causes a prerequisite failure, not silent case removal.

This is a logged implementation correction for the no-feedback information boundary,
not an outcome-driven method selection or a claim of an optimal fixed plan.

## Acquired evidence encoding and hardware batching

The fixed 64-wide deterministic event bins include public port direction and
receive/send indicators as well as values, local time differences, missingness,
repeat-ID statistics and acquired-only matching residuals. Opaque ID magnitudes,
clock ordinals, simulator ticks, labels and actual offsets are never numerical
features. The preprocessing version is `cgx.acquired.v2.64bin`; caches are keyed by
both acquired-history hash and feature-source hash.

B200 profiling supports a default microbatch of 128 for the registered batch of 128.
`CGX_MICROBATCH` may lower memory use by accumulating the same batch before one
optimizer update. This is a hardware setting; it changes neither examples per update
nor update caps. Source preprocessing, cached GPU updates, validation simulation and
online decision latency are timed separately. The capacity profile is technical data,
not a scientific estimate of diagnostic quality.

A fixed plan's individual choices each receive the same two-second selection limit
as an adaptive choice. Constructing up to six choices before any new evidence does
not compress six decision limits into one two-second total limit.

Formal training/loaders enable strict PyTorch deterministic algorithms, a fixed cuBLAS
workspace configuration, deterministic cuDNN, disabled benchmarking and disabled TF32.
Unsupported deterministic operations fail explicitly. Capacity profiling is repeated with
these settings before sealing; floating-point invariance tolerances remain as specified.
