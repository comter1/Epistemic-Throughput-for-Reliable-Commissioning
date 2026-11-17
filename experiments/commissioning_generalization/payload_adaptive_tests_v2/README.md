# CGX-PAYLOAD-ADAPTIVE-TESTS-V2

An isolated, simulation-only extension of the read-only payload_change_v1 experiment.
Six next-test policies share the same 17-coordinate payload/bias estimator, frozen
Panda torque controller, motion screening, parameter updates, speed choice, stopping,
public verification, and eight independent private audits. The primary contrast is
repeated work-decision selection versus planning once after the common B0 measurement.

The complete user specification is in documents/REQUEST.md. No GPU training,
real robot connection, or changed controller gains are involved. The current run is:

`/work/Users/leiyo/cgx_data_v2/payload_adaptive_tests_v2/20261117T212456Z`

## Stages

From experiments/commissioning_generalization, with PYTHONPATH=.:src and one BLAS
thread, run each stage with `python -m payload_adaptive_tests_v2.runner --stage STAGE
--run-root RUN_ROOT`: audit, qualify-source, fit, branch-checks, validate, seal,
evaluate, report, package. Use a new external directory for a fresh reproduction.
The pinned V1 archive and Panda assets must be available at the declared inherited
paths. The runner dispatches physical work only to SSH ports 2072 and 2379.

Initial profiling uses eight workers and 20 source blocks. Normal dispatch requests
136 workers on 2072 and 44 on 2379; workers clamp this to their actual cgroup CPU
allocation and record memory/CPU limits. Every physical/public-model integration
reserves budget before execution. Interrupted reservations remain conservatively
charged. No test runs start without the sealed code, asset and configuration hashes.

## Models and boundaries

PublicPredictionModel accepts only public geometry, controller configuration,
proposed reference and explicit parameter hypotheses. It integrates nominal dynamics
at 500 Hz with 100 Hz sampled control. MuJoCo transition/inverse-dynamics derivatives
are refreshed at 20 Hz; parameter and feedback sensitivities propagate at every
physics step. It retains nominal tracking error. Source-held-out margins cover
unmodeled tracking, torque and clearance residuals. Its confidence set is an
engineering approximation, not a safety certificate. Actual physical jobs decide
verification/audit outcomes.

The source estimator fit uses object indices 0..5; 6..7 are disjoint-object margin
and uncertainty calibration cases, with all contexts kept together. Structural
family membership is inherited exactly, with fresh numerical objects. Validation
runs all three inflation multipliers and all three fixed-plan templates. Test
statistics average noise seeds, then bootstrap families and objects while retaining
all contexts of an object. The test matrix is 64 objects x 3 contexts x 3 seeds x
6 methods, plus 576 evaluator reference groups.

Source-only cloned continuations include integration, routing, bias, AR and RNG
state. Clones are never a deployed action. New future noise is paired across candidate
branches, without altering the already acquired prefix. Synthetic observations are
used only by the explicit conditional-Gaussian mechanism test.

## Timing and artifacts

Identification includes B0, entry, dwell and return. A 0.75-speed block is rounded
up to a whole 100 Hz sample (10.67 s) and charged at that duration. Final release
requires one public verification; half-speed consumes the whole 20-second allowance.
The primary serial metric adds measured fitting/planning/update CPU to physical
time, as requested. Supplemental per-decision and component wall times are retained.
Shared exact public-model caches are available to all policies; actual cache state
is recorded, so measured CPU latency is not presented as a standardized cold-start
benchmark. Physical measurement records are never synthesized from predictions.

Large raw arrays remain external. The analysis ZIP includes complete per-run/step
summaries, frozen configs and manifests, code, audit evidence, figures, representative
100 Hz/500 Hz traces and an exact SHA256 reference index. The index is not the raw
arrays. Engineering revisions, failed attempts and known unsaved tails are explicitly
retained and documented. The final ZIP must remain below 500,000,000 bytes.
