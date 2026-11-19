# Implementation audit — CGX-PAYLOAD-COMMISSIONING-V1

Audit scope: existing repository implementation and pinned Panda XML, before new payload simulations. Base commit `65a55004a0baf1cca6ac2977d3025c0acbdd736c`. New work is isolated under `experiments/commissioning_generalization/payload_change_v1/`; older experiments and tables remain protected inputs.

## Existing controller and actuator path

`src/cgx/env/system.py:252–265` routes measured state through the healthy software graph, then computes `q_cmd = q_meas + 0.20*(q_ref-q_meas) - 0.02*dq_meas`, clips it to joint range, and writes it into `data.ctrl[:7]`. This is a position command, not a direct torque request. `panda.xml:264–273` defines affine general actuators: gains 4500/4500/3500/3500/2000/2000/2000 with negative position and velocity bias gains. These implement an inner position/velocity servo. Force ranges are ±87 Nm for joints 1–4 and ±12 Nm for joints 5–7; these are asset limits, not a manufacturer specification inferred by this experiment. The gripper actuator is separately retained.

No controller load configuration or inverse-dynamics call exists in this execution path. Text scanning the old environment sources and pinned Panda XML found no `qfrc_bias`, `qacc`, `gravcomp`, `mj_inverse`, or mass-matrix access. This is an absence finding for those inspected files; it does not make the old position actuator a load-aware torque controller. The XML contains a finger coupling equality, not an ideal arm position equality. General actuators have finite force ranges.

Consequently, merely storing a payload-setting dictionary in the old controller would not implement the proposed intervention. The new namespace needs a separate fixed torque controller whose inverse-dynamics model contains only public configured load values. Controller, estimator and private plant models must be independent compiled objects. The detailed new decision, gains, limits and source qualification results belong in `CONTROLLER_DECISION.md` and the executed checks; they are not declared passed here.

## Existing physics, reference and software observations

`System.run` executes 1000 software ticks of 0.01 s, with five `mj_step` integrations per tick; `_load_model` sets timestep 0.002 s. The original task generator (`system.py:35–41`) draws three joint targets around HOME, with amplitudes [0.14,0.10,0.14,0.10,0.12,0.10,0.14] rad and deadlines [3.2,6.4,9.6] s. `reference_at` uses 0.1 s cubic smoothstep transitions beginning at 0.5,3.2,6.4 s. L0 must preserve this exact reference even if the new torque/rate constraints fail it.

All software graph nodes are actually evaluated in topological order. `system.py:229–275` records receive/send events at graph input/output ports. Cache, queue, adapter, splitter and merger semantics are implemented by `env/modules.py`; with healthy settings they retain their public representation/routing roles. The graph will be chosen by source hash without looking at payload outcomes by the orchestration code; this audit does not select a favorable graph.

The old sensor observes q and dq with standard deviations 1e-4 rad and 1e-3 rad/s (`system.py:169–170,235–237`). Port records contain port, local timestamp, opaque message token, event type, payload values and missing flag (`system.py:212–215`). There is no joint torque sensor in the old public record. The old external stream uses simulator hand position plus independent noise and a separate reference FK, at 10 Hz. The new experiment declares encoder-based FK and adds synchronized 100 Hz torque/reference/configuration/flags channels, preserving all software port logs rather than selectively recording interfaces.

Old `System.run` resets state to HOME each call (`system.py:161–167`); that API cannot be reused unchanged for consecutive commissioning test blocks, where movement, return and settling are charged and physical state must continue. New block execution requires a continuous private plant object. The original old evaluator remains unchanged.

## New public/private boundary to verify

The backend may use private mass/inertia to integrate actual motion and may define simulated execution-torque measurement as `qfrc_actuator` plus the declared persistent bias and AR(1) sensor error. This is an ideal-actuator simulation sensor, not a claim of matching the Panda link-side torque sensor. Requested torque, limited/applied command and measured torque must be distinct fields. The controller/estimator must not receive private qacc, bias force, mass matrix, material density, true payload, group label, private audit results or a reference to the private plant object.

A source-fitted ten-parameter inertial estimator and seven persistent joint sensor bias states are required. Measured velocity, a frozen filter, and actual settled-state checks determine acceleration/static observations. Source-only physical consistency, duplicate-event invariance, rank, shared-bias, frame/sign and regressor reconstruction checks must be executed. Test data cannot tune gains, noise, priors, filters or stopping thresholds.

## Protection and regression status

`checks/protected_before.json` records actual protected-file hashes; `checks/protected_after.json` must compare them at completion. The snapshot records its coverage explicitly: tracked old repository files, pinned Panda assets, prior protocol/model/result summaries and existing preservation inventories. Large raw execution stores are referenced by their prior immutable inventories rather than being silently presented as freshly rehashed.

Old simulator regressions will run as a few executed healthy/cache/queue/adapter cases on authorized remote CPU resources, with their simulated time included in this experiment ledger. At this initial audit they are **pending**, not passed. New source qualification and all formal source/validation/test results are likewise pending.

## Executed preservation baseline and old-controller regressions

The initial preservation snapshot completed with 729 files; see `checks/protected_before.json`. On authorized port 2982, seven original simulator runs (70 simulated seconds, charged to the new experiment physics ledger) passed: nominal healthy task, actual cache/queue/adapter perturbations, and exact restoration to the healthy physical trajectory after the addressed repair. `checks/legacy_regression.json` contains per-case physical differences and metrics. This verifies old behavior only; it is not a claim that the new payload controller has passed qualification.

## Final source engineering decisions and scope (before test sealing)

The first source engineering attempt is retained under `engineering_attempt_01`. It exposed 167 L0 hard aborts under the initial high PD gains; these are not excluded outcomes from a final formal test. Source traces identified nonlinear overshoot caused by high feedback demands interacting with the unchanged 1000 Nm/s command-rate limiter. A separate source restart bug recreated torque-sensor bias after an abort; the final source collector preserves the same case-persistent bias and AR state when a declared source-only physical restart is necessary. All failed integrations remain charged and archived.

Source-only nominal gain qualification selected `payload-id-pd-v2-source-qualified`, Kp=[200,200,150,150,80,80,60] and Kd=[10,10,8,8,5,5,4]. All 192 source payloads under their true configured loads had zero L0 hard aborts. Only 59/192 passed every L0 tracking/saturation acceptance requirement, while 192/192 passed the declared smooth work trajectory. The exact original L0 cubic100ms transitions remain unchanged. The final10s work trajectories are smooth motion specifications covering near translation, farther reach and tool orientation; their `speed=1.0` is relative to that declared workload. Passing them does **not** establish restoration of the original100ms transitions under all acceptance constraints. Final source effect comparisons are in `source_validation/payload_effects_final`, with raw initial-gain comparisons separately retained.

The gripper is fixed at its original0.04m open position and retained as known geometry/mass exactly once. Its finger joints/tendon/actuator are removed only in the new payload model; this is a declared controller/model change, not an unchanged reproduction of old software-fault numerics. Full equations, mapping and limits are documented in `CONTROLLER_DECISION.md`.

The simulated execution-torque sample at a100Hz endpoint reflects the preceding held command. Source engineering therefore aligned it with midpoint measured q/dq and the backward measured-velocity difference over that same interval, followed by frozen Savitzky–Golay11 filtering. No hidden qacc or commanded acceleration is substituted for measured acceleration. The initially misaligned derivative result and the correction evidence remain in `checks/measured_interval_alignment.json` and the archived profile.

Source calibration selected covariance inflation12.281906827006292 from observed errors, with uncertainty checked again independently on validation/test. The stopping envelope is the maximum full torque width over a joint10D approximate99% Gaussian parameter ellipsoid (radius4.8175980694693035), normalized by actuator limits. It is **not** a collection of pointwise99% torque marginals. Reported scalar mass and COM coordinate intervals separately use z=2.5758293035489004; intersecting the three coordinate intervals is not a simultaneous99% COM-region claim. `checks/interval_semantics_decision.json` records this pre-validation interpretation decision.

Validation evaluated64 cases with seed7, four methods and three shared stopping tolerances. The selected common threshold0.10 yielded256/256 correct releases and0/256 false releases at20s mean online physical time. Threshold0.05 yielded255/256, and0.02 yielded192/256; both had zero observed false releases. This is an engineering threshold selection using independent four-trajectory audits per configuration, not a safety certificate or test result.

Five original finite-delay benchmark unit tests were executed remotely on port2379 and passed; they use a finite probability model and no MuJoCo integration, so they add zero physical simulation seconds. The exact output is `checks/legacy_five_unit_tests.log`. Together with the seven charged old Panda regression executions, this documents both legacy regression paths.
