# Codex execution plan: choose the next robot test from the measurements already obtained

**Experiment ID:** `CGX-PAYLOAD-ADAPTIVE-TESTS-V2`  
**Suggested branch:** `exp/payload-adaptive-tests-v2`  
**New code directory:** `experiments/commissioning_generalization/payload_adaptive_tests_v2/`  
**Parent experiment:** `CGX-PAYLOAD-COMMISSIONING-V1`  
**Deliverable:** completed simulations, per-case results, source code, audit records, figures, and one analysis ZIP.  
**Execution scope:** simulation only; no connection to a real robot.  
**Main question:** after the first measurement, does choosing subsequent test motions again reduce the time needed to approve a specified job, compared with a good fixed test plan?

## 0. Read this first

Implement and execute this experiment on top of the existing Panda/MuJoCo project. This document is an execution instruction, not a request for another proposal.

All methods may record all existing software interfaces and public sensors. All methods use the same estimator, controller, parameter-update rule, motion screening, verification, and stopping rule. Only the rule that selects the next test motion changes.

The experiment has two outputs that must be kept separate:

1. **What happened physically:** measurements, tracking errors, torque requests, contacts, accepted jobs, and elapsed time.
2. **What the selection model predicted:** parameter intervals, admissible tests, predicted job margins, and the estimated value of another test.

A change in the second output does not replace measurement of the first.

Complete the formal comparison even if a fixed plan wins. Engineering checks may stop execution for invalid physics, leaked private fields, or broken accounting. A lack of adaptive improvement is a result, not a reason to change the test set or requirements.

### Decisions already made for this round

- Keep the seven-joint Panda and the V1 torque controller.
- Keep all logs available. Do not add an interface-count or byte restriction.
- Keep the V1 payload range for the main experiment. Do not silently enlarge masses or lower hardware limits to create a difference.
- Add three public workstation/job contexts, smooth selectable test motions, and a common uncertainty-aware motion screen.
- Let fixed methods use current measurements for controller updates, motion screening, speed selection, and early stopping.
- Compare fixed ordering, planning once after an initial measurement, a hand-written rule, repeated variance-based selection, and repeated decision-based selection.
- Do not train a new GNN, Transformer, RL controller, or language model in this round.
- Measure whether later **observed values**, not merely public workstation identity, change a useful next-test choice.

---

## 1. What the existing implementation actually established

Reopen the actual parent run before editing code:

- `Payload_Change_Commissioning_20261116T180000Z_GPT_Pro.zip`;
- `payload_change_v1/code/payload_change_v1/design.py`;
- `payload_change_v1/code/payload_change_v1/estimator.py`;
- `payload_change_v1/code/payload_change_v1/physics.py`;
- `payload_change_v1/code/payload_change_v1/trajectories.py`;
- `payload_change_v1/code/payload_change_v1/commissioning.py`;
- the parent's frozen protocol, `configs/`, `CONTROLLER_DECISION.md`, and `results/`.

The source archive was inspected when preparing this plan. Relevant facts are:

| Parent implementation | Consequence for this experiment |
|---|---|
| Fixed four-pose, fixed mixed, and adaptive methods had identical primary aggregate results: 576/576 successful releases, 10.125 s identification, 10 s verification, 38 parameter updates | More inspection was not an established adaptive advantage |
| `build_design_bank` computes `information17` from public reference position, velocity, and acceleration grids | Much of the expected information can be calculated before new sensor values arrive |
| The estimator maintains ten payload inertial parameters and seven persistent torque biases | Preserve their joint update rather than resetting biases each block |
| `choose_action` uses expected reduction of discrete work-decision entropy, but switches to task-torque variance reduction when that entropy is at most 0.01 bit | Record which branch is actually used; a variance-only branch may have little dependence on measurement values |
| For the linear Gaussian prospective model, posterior covariance is independent of the newly observed numerical values, conditional on the design | A tighter covariance threshold alone need not create different test choices |
| V1's work assessment used a 5% torque-impact rule, a 95% torque-limit rule, and a validation-selected interval threshold | These are not a direct prediction of whether a specified position-error requirement will pass |
| V1 L0 preserves 100 ms reference transitions; many correctly configured cases violate its trajectory/saturation acceptance, even though the smooth final work passes | Do not force L0 to be the mandatory first motion in a constrained commissioning experiment |
| The three main V1 methods and the existing-settings reference all passed the declared final smooth work | Separate easy-to-approve cases from cases in which a configuration or speed decision actually matters |

These facts motivate the new design. They do not establish that the new selector will win.

Write `PARENT_AUDIT.md` with exact function names, commit, file hashes, and the facts above verified against the repository. If a local file differs from the archive, record both versions and use the verified parent scientific implementation.

---

## 2. Physical scenario and the question being tested

A robot holds a rigid part whose exterior and mounting frame are known. Its mass distribution may differ from the controller's configured load. The robot must move the part to a declared workstation and finish at a declared pose. It can conduct short tests before approving the job.

The robot knows:

- its arm/tool model and configured load;
- the part's exterior dimensions and mount;
- the workstation fixtures and nominal work path;
- the job's time, position-error, torque, joint, and contact requirements;
- measurements it has actually collected.

It does not know the actual mass distribution, true inertial parameters, sensor-bias draw, future noise, or private audit outcomes.

The possible benefit of feedback is concrete: readings can change which load explanations remain plausible, which motions pass the common screen, and which measurement would separate competing work decisions. A public difference between two workstation geometries is **not**, by itself, evidence that reading new measurements and replanning is useful.

### Literature basis, not imported performance claims

- Zhang, Zhou, and Vasudevan (RSS 2025) study online payload identification while respecting robot and obstacle constraints. Use the scenario as motivation; this plan does not reproduce their robust controller or their formal safety guarantee. [E1]
- Cooper et al. (T-RO 2021) compare batch and sequential experimental designs on real vehicles. This motivates a direct fixed-versus-feedback comparison. [E2]
- Universal Robots documents a four-pose payload/CoG procedure. Include a strong fixed multi-pose design, not only random motions. The UR procedure is not a Panda certification procedure. [E3]

---

## 3. Freeze the inherited physics and sensors

Use the V1 final controller `payload-id-pd-v2-source-qualified`:

```text
Kp = [200, 200, 150, 150, 80, 80, 60] Nm/rad
Kd = [10, 10, 8, 8, 5, 5, 4] Nm*s/rad
physics step = 0.002 s
sensor / control / software routing = 100 Hz
joint torque limits = [87, 87, 87, 87, 12, 12, 12] Nm
command slew limit = 1000 Nm/s
```

The torque limits come from the pinned simulation model. Other numerical requirements in this document are experimental settings, not manufacturer guarantees.

Retain:

- the rigidly mounted payload and the known fixed gripper;
- the executed healthy software graph;
- separately compiled plant, configured controller, and public estimator models;
- q/dq measurement noise from V1;
- torque noise: 0.05 Nm stationary standard deviation, AR(1) correlation 0.6;
- a persistent per-joint torque bias, standard deviation 0.02 Nm;
- continuing noise/bias and physical state across test blocks;
- full software/sensor recording at the inherited rates;
- private 500 Hz evaluation, not visible to the selector.

The controller uses configured parameters only. Changing a public configuration never changes the physical object's mass or inertia. The estimator never reads plant `qacc`, `qfrc_bias`, mass matrices, or private sensor errors.

No teleportation to HOME between actions. Motions must include transitions and return travel. Parameter changes at rest are common to all methods and are logged. Do not turn an offline state clone into a deployable reset action.

If the frozen controller cannot track a new reference with the declared known-parameter model, first exclude references that violate the public kinematic envelope. Any further controller change requires a separate, explicitly named experiment, not a silent edit during V2.

---

## 4. Payloads, contexts, and data splits

### 4.1 Payload generation

Reuse the physically consistent box-and-internal-weights generator. Keep the main mass support at 0.2--1.5 kg and use the same geometric construction, declared exterior dimensions, and physical inertia checks.

Keep all four change groups:

- G0: unchanged load, correct settings;
- G1: changed load, settings correctly updated;
- G2: mass changed, old settings retained;
- G3: mass distribution/CoM changed, old settings retained.

Group and internal-family labels are evaluator-only. All groups remain in the result denominators.

Use fresh numerical case seeds and the parent's disjoint structural family split. Do not move families across splits to improve outcomes. Do not filter on stale-setting failure or on the winner of a method comparison.

### 4.2 Three public work contexts

Construct exactly three work-context templates using public geometry and kinematics:

| Context | Work and fixtures | Purpose |
|---|---|---|
| W0 | Open work area, near transfer, terminal pose | Reference without added fixtures |
| W1 | Transfer to a farther, laterally offset jig; a known side fixture | Changes useful arm posture and available motion directions |
| W2 | Transfer with a tool-orientation change near a known overhead/side fixture | Changes wrist motion and clearance requirements |

Generate Cartesian endpoints from predeclared joint-space seeds within the nominal reachable workspace. Use simple box fixtures, represented in the actual MuJoCo plant as well as the public model. A deterministic kinematic planner must provide a collision-free nominal approach and return path.

Specify fixture dimensions and placements in `configs/workstations.json`. The construction may use only public geometry and nominal kinematics, never true test payloads or method outcomes. Default minimum nominal clearance is 30 mm for W1 and W2. Do not shrink this clearance after viewing the comparison. It is acceptable for some test patterns to be unavailable in a context.

Within a context, different hidden loads must share the same public job and fixtures. The public endpoint/path seed is independent of hidden payload group and severity. A fixed method receives this context and may generate a context-specific plan.

If an exact Cartesian request is kinematically unreachable, log that proposal before applying a deterministic public-geometry resampling rule. A physical failure after case creation is not an excuse to resample the case.

### 4.3 Main matrix

| Split | Hidden payload families | Objects per family | Different objects | Contexts per object | Object-context cases | Sensor seeds |
|---|---:|---:|---:|---:|---:|---|
| Source | 12 | 8 | 96 | 3 | 288 | 5 |
| Validation | 4 | 8 | 32 | 3 | 96 | 7 |
| Test | 8 | 8 | 64 | 3 | 192 | 11, 23, 37 |

Every family balances the four change groups. Thus each main method has 576 test runs, but only 64 different objects and 192 object-context cases.

Keep all three contexts of the same object together in statistical resampling. Do not report 576 independent robots.

Store separate public and private manifests. Freeze their hashes before evaluating selectors on test outcomes. No public numeric feature may encode a hidden family/group/case identifier.

---

## 5. Declare the work requirements before selection experiments

The new task is a **simulated transfer-and-alignment job**, not insertion/contact and not a claim about a particular factory.

Use this primary work specification:

| Quantity | Requirement |
|---|---:|
| End-effector path RMS position error | <= 5 mm |
| Maximum path position error | <= 10 mm |
| Position error throughout final hold | <= 2 mm |
| Final hold duration | >= 0.5 s |
| Unexpected fixture/self contact | 0 |
| Joint limit violation | 0 |
| Maximum continuous saturation for one joint | <= 0.1 s |
| Fraction of task time with any saturated joint | <= 1% |
| Hard abort | 0 |

Use the original reference velocity limit of 1.5 rad/s and acceleration limit of 3 rad/s^2. Keep the inherited physical limits; do not lower torque limits to create a selection problem. Record requested as well as applied torque, so command clipping does not make an over-demanding motion appear feasible.

The normal job uses 10 s of nominal motion. The permitted slow job follows the same geometry at 0.5 speed and takes 20 s. All methods receive the same public representative work path and the same declared distribution of small path variations.

These precision values define this new simulated work requirement. They are not estimated from the test errors and are not manufacturer accuracy ratings. Also report continuous errors and a fixed descriptive threshold sweep, including the old 20/50/20 mm specification, without retuning methods on that sweep.

### Characterize the job rather than assume it is difficult

An evaluator-only reference runs each case with:

1. original configured parameters, normal speed;
2. true parameters, normal speed;
3. true parameters, half speed.

Report counts in the following groups:

- original settings already pass;
- original settings fail, true settings at normal speed pass;
- true settings require half speed;
- neither tested speed passes even with true settings.

These groups describe outcomes; they do not select which cases remain in the experiment. The fourth group means failure of these controller/configuration/speed choices, not impossibility under every controller.

---

## 6. Test motions: all methods can collect everything

### 6.1 Common first test B0

Replace mandatory L0 by one short, smooth, source-qualified initial test, identical for all selection methods:

```text
HOME hold: 2 s
smooth wrist tilt: 2 s, default delta joint 5 = +0.15 rad, joint 6 = -0.12 rad
hold the tilted pose: 1 s
smooth return to HOME: 2 s
settled HOME hold: 1 s
Total: 8 s
```

Use quintic transitions with zero endpoint velocity and acceleration. The same initial public HOME target and noisy measurements are used for all methods. Screen the full motion against the declared initial payload range and public geometry.

If this exact B0 is geometrically unavailable in a context, use a predefined mirrored-tilt alternative; if neither is available, use an 8 s HOME hold. Select the alternative using public geometry and the initial common parameter set, not the hidden load. All methods in that context get the same alternative.

L0 remains a reported parent result. Do not represent B0 as an identical rerun of L0, and do not describe cross-experiment accuracy changes as a controlled selector comparison.

### 6.2 Candidate library

Start with the six V1 static patterns S1--S6 and four dynamic patterns D1--D4. Each pattern includes approach, hold/excitation, return, and settling.

Provide time scales 1.0, 0.75, and 0.5:

- positions follow the same path;
- velocities scale by s;
- accelerations scale by s^2;
- duration scales by 1/s;
- sensor sampling remains at 100 Hz.

Use a shared wrapper that selects the fastest version passing the current common motion screen. A method selects a **base pattern**, not a privileged speed unavailable to fixed methods. Record the actual version and duration.

Repetition is permitted and charged. A genuinely new repeat generates new measurements while retaining persistent biases. Prevent a zero-duration or metadata-only loop.

For each context, remove only patterns whose nominal geometry is impossible; retain patterns that can become available after load uncertainty decreases. A currently blocked pattern can be revisited later. Compute from the measured start estimate, not an assumed perfectly reached HOME state.

### 6.3 Budgets

- Identification-motion budget: 60 s, including B0, travel, holds, and returns.
- Physical public verification: at most 20 s.
- Maximum additional test blocks after B0: 6.
- Full software and sensor recording stays available in every block.
- No extra uncharged startup trial, calibration, return move, or public verification is allowed.
- Parameter fitting and test planning CPU time is measured and added when reporting serial deployment time.

---

## 7. One common estimator and one common controller-update rule

Reuse the V1 17-variable estimator: ten inertial parameters plus seven persistent joint-torque biases. Preserve measurement identity, noise whitening, bias continuity, physical consistency projection, and uncertainty inflation.

Refit sensor/residual calibration and uncertainty inflation on the new **source** trajectories, with one shared fitted result for all methods. Use the same public physical measurements, not private noise/bias draws. Do not fit one estimator per selector.

Source collection must cover static, dynamic, slow, and normal-speed motions using a mixture of geometry-feasible fixed and randomized patterns. Keep failed and aborted source attempts in the ledger. The estimator cannot use records beyond the actual abort.

Write predictive-coverage tables by motion kind, speed, and context. Check that the intervals are not merely tiny in the direction chosen by the current method. Projecting a Gaussian mean into the physical set is not, by itself, proof of coverage.

### Common parameter update

At each rest boundary, every method may update its configured load using the same projected posterior estimate, provided the common screen permits the next operation under that configuration. Record old/new values and the operation time.

Remove V1's fixed 5%-of-torque impact rule as the primary reason to retain a wrong setting. Decide whether to keep or update by predicted job requirements and the common verification rule below.

Retain the original configured load as a candidate for the final job, even after intermediate testing updates. All methods receive the same update opportunities. No hidden true parameters enter updates.

The new experiment therefore compares motion selection while sharing parameter estimation and control adaptation; it does not give parameter updates only to the adaptive selector.

---

## 8. Predict whether motions fit the limits, using only public models

Create a `PublicPredictionModel` that receives public robot/fixture models, a proposed configuration, a candidate reference, and a parameter hypothesis. It has no pointer to the unknown plant.

### 8.1 Fast common prediction

Implement a first-order closed-loop tracking calculation around the current public configured model, not just a torque-percentage check.

A suitable implementation is:

1. Integrate a nominal configured model under the frozen controller along the candidate reference.
2. Compute payload torque regressors along that path.
3. Propagate parameter perturbations through the linearized closed-loop tracking dynamics, including the controller's sampled update rate. Retain the nominal tracking error.
4. Map joint tracking deviations to end-effector error and all relevant link/fixture clearances using public Jacobians and geometry.
5. Derive requested-torque, tracking-error, and clearance predictions for the current joint parameter distribution.

For a scalar linearized output g, the usual ellipsoid calculation is:

    predicted upper = g(mean) + radius * sqrt(J_g * covariance * J_g.T) + source_margin

Use the joint parameter ellipsoid, not a separate per-timestamp 99% interval claimed to hold simultaneously. Include the parameter-dependent nominal mean, not only covariance.

Choose source margins from a held-out part of source data, measuring the difference between predicted and actual maximum tracking errors, torque requests, and clearances. Margins must be shared across methods and sealed. The approximation must be checked where the slew limiter or saturation is active. Mark those cases unsupported by the linear prediction unless a declared secondary public-model check resolves them.

### 8.2 Common motion screen

A candidate is eligible only when the shared predicted bounds satisfy:

- configured joint/rate/reference limits;
- requested-torque limits with the sealed source margin;
- a tracked geometric clearance larger than the sealed tracking margin;
- enough remaining time to execute and return;
- no unresolved residual/model-consistency alarm for that motion class.

Bounds are computed under the same current parameter estimate for **all** methods. Fixed ordering may skip currently blocked candidates, try later candidates, and revisit deferred candidates after new data. It may not be forced to execute a motion the adaptive method would reject.

A Gaussian/empirically inflated interval plus a finite prediction grid is a model-based screen, not a formal hardware safety certificate. Report actual contacts, saturation, and aborts separately. [E1] uses different assumptions and a different robust controller.

### 8.3 Secondary public-model check

When the linear prediction is outside its source-validated range, run a limited check using separately compiled hypothetical payload models sampled from the public estimate. Never clone the unknown plant for online prediction.

Default: posterior mean plus 16 deterministic, physically valid parameter hypotheses, with cached public models and the actual frozen controller. Verify the chosen configuration/motion, not every possible branch of the future investigation. If results conflict, reject the candidate or keep the case unresolved; do not use the private plant to resolve the conflict.

This check is approximate and its CPU/simulation cost is recorded. All methods use the same check and cache contract. Set a maximum of 10 s of serial online prediction/planning time per decision; a timeout is an unresolved planning step or the common frozen fallback, not an unrecorded extra computation.

### 8.4 Real execution remains authoritative

Every selected action is executed by the actual hidden-load MuJoCo plant. A common public-sensor watchdog may interrupt. Private exact-state evaluation may terminate a simulated physical violation and records it as a violation. Private values never alter the selector's preferred action or posterior.

Do not replace actual validation by an inverse-dynamics or tracking prediction.

---

## 9. One common rule for updating, slowing, stopping, and releasing

At every rest boundary consider the same finite deployment choices:

- initial configured load, normal speed;
- current/estimated load, normal speed;
- initial configured load, half speed;
- current/estimated load, half speed;
- unresolved: do not approve.

Duplicate configurations are collapsed. Values are explicitly saved.

Apply the predicted work bounds from Section 8 to the full representative job, using the requirements in Section 5. This replaces the old condition "torque uncertainty is below 10%" as the primary readiness rule.

Among eligible choices prefer:

1. normal speed;
2. keeping an already adequate configuration;
3. otherwise the shared estimated configuration;
4. half speed only when no normal-speed choice is supported.

A positive prediction is an instruction to perform public verification, not approval by itself. Execute the same public job verification, with full logging. Approve only if the common prediction/consistency conditions and actual public verification pass. If no choice passes, continue identification while time remains; otherwise hold unresolved.

The verification allowance is shared: one normal-speed 10 s trial may be followed by a second 10 s normal trial only if the remaining budget permits and this retry rule was sealed. A half-speed trial takes the full 20 s. There is no uncharged attempt at both speeds. Primary default is one verification attempt; include verification evidence only after it was physically obtained, and do not enter a hidden retry loop.

After release, execute eight private audit jobs at the released speed, with independently generated path variations inside the declared job distribution. These results are not returned to the method. For unreleased cases, evaluator-only audit of the last proposed configuration may characterize missed opportunities but cannot count as a successful deployment.

All fixed methods can stop early under exactly this rule. Record whether a result difference is caused by test order, common screening, parameter updates, stopping, or verification.

---

## 10. Methods to run

Use these names verbatim in reports, together with the common estimator/controller identification.

| ID | Display name | What can change after B0? |
|---|---|---|
| `b0_verify_only` | Initial Test Only | Common estimation/update/verification; no additional identification motion |
| `fixed_prior_plan` | Prior-Designed Fixed Order | Shared gate, speed, settings, and stop; test priority order remains fixed |
| `plan_after_b0` | Plan Once After Initial Measurements | Plan computed after B0; shared gate/settings/stop continue, ranking is not recomputed |
| `simple_parameter_rule` | Hand-Written Parameter Rule | Uses current uncertainty and work sensitivity with a sealed simple rule |
| `adaptive_variance` | Rechoose for Work-Torque Variance | Recomputes variance reduction per second after each block |
| `adaptive_work_decision` | Rechoose for Work Decisions | Recomputes which test separates competing work decisions after each block |

Also run `true_payload_reference`, evaluator-only, without presenting it as a deployable selector.

### 10.1 Prior-Designed Fixed Order: make it strong

Give it all public context, configured parameters, and the source prior. Generate a context-specific order before new measurements using a bounded search over the same motion library. Include three predeclared candidate templates:

- four complementary static patterns, selected using the source prior;
- static plus dynamic patterns optimized for the declared work sensitivities;
- a fixed order optimized using the **same decision score as the adaptive method**, evaluated without future sensor outcomes.

Use a beam width of 3 and depth at most 4, then append uncovered patterns using the same frozen rule. All time costs and shared screening apply. A compound fixed test may remove unnecessary intermediate return-to-HOME segments only if the exact same path-composition option is provided to the other planners and executed in the cost accounting.

Select one template on validation by: violations/false releases first, then correct normal-speed release coverage, then capped total time. Publish all three validation outcomes. Do not select using the test suite.

Its live shared gate sees current measurements, but it never changes the relative priority order on that basis. Call it a fixed-order method with a shared feedback screen, not a controller that ignores all feedback.

### 10.2 Plan Once After Initial Measurements

After the common B0, use its actual estimate and the same bounded search to produce the remaining priority order. Later forecasts used while building this plan use expected zero-innovation updates; the planner cannot see actual future readings.

Keep the order fixed thereafter, while allowing the same live gate, fastest-eligible speed rule, configuration updates, and early stopping. This is the principal strong comparator for the additional value of **later** readings.

### 10.3 Hand-Written Parameter Rule

Compute how uncertainty in mass/first moments and in inertia contributes to the declared work-torque uncertainty. Normalize contributions in work units; do not compare raw kg and kg*m^2 variances.

- If mass/first-moment uncertainty dominates, select the next eligible static pattern from a source-designed diverse order.
- Otherwise select the next eligible dynamic pattern from a source-designed order.
- Use the same live screen, speed wrapper, and stopping rule.

The class choice uses the current estimate. The within-class order is frozen. Record ties and tie-breaking by public action identity, not hidden case ID.

### 10.4 Rechoose for Work-Torque Variance

Port the V1 covariance-based score, using current work sensitivities and actual available motions:

    score(a) = [current work-torque variance - expected variance after a]
               / charged physical duration(a)

Keep persistent bias and uncertainty inflation in hypothetical updates. This method identifies whether an advantage comes from screening and covariance bookkeeping without needing measurement-dependent work decisions.

### 10.5 Rechoose for Work Decisions

Use expected reduction in uncertainty about the feasible deployment choices, divided by actual motion time. Define the target carefully so it does not change halfway through an expected-information calculation.

At current history h:

1. Freeze the current candidate configuration set D_h from Section 9 for this one decision calculation.
2. For a hypothetical physical parameter vector theta, use the public work model to compute the binary pass/fail vector

       Z_h(theta) = [pass(keep, normal), pass(estimate_h, normal),
                    pass(keep, half), pass(estimate_h, half)].

3. Use the **same** D_h, job and definition of Z_h before and after each hypothetical observation. The estimated configuration cannot silently change inside only one side of the entropy difference.
4. Generate future readings or sufficient-statistic outcomes from the joint payload-plus-bias measurement model. Do not independently reset bias or reuse a future actual reading.
5. Compute

       score(a | h) = [H(Z_h | h) - E_o H(Z_h | h, a, o)] / duration(a).

6. After a real action returns measurements, recompute the actual posterior, current deployment proposals, screening, and the next decision.

This is an approximate information score for the public work model, not measured true Shannon information in the physical simulator.

Start with 128 current parameter samples and 16 hypothetical outcomes, reusing deterministic Monte Carlo draws across candidates to reduce ranking noise. Check candidate rankings on source with twice these counts. Save mean score, Monte Carlo variability, feasibility and expected remaining decision ambiguity for each candidate.

The fast work predictor is used in the inner loop. Secondary public-model simulations from Section 8 are limited to ambiguous/unsupported checks, not repeated thousands of times per candidate.

When all pass/fail vectors agree and a release is supported, stop and verify. When the decision score is tied but release is still blocked by residual/model inconsistency, use one declared common fallback: the source-designed diverse order with the current gate. Record `fallback_reason`; do not call a run that always follows this order evidence-adaptive success.

---

## 11. Source-only tests that explain when feedback can matter

These are part of the deliverable. They do not select favorable test cases.

### 11.1 Check the outcome-independent special case

For a fixed design matrix, noise law, and prior covariance, generate two different synthetic observation vectors. Verify that the unprojected Gaussian posterior covariances agree while means differ. Then show whether the work-decision score and common motion screen differ.

This prevents attributing an unchanged covariance design to failure of reinforcement learning. It also checks that new measurement values actually have a route to influence the intended decision.

### 11.2 Predictive action maps

Use a fixed hash-selected set of 24 source object-context histories after B0: eight per context, balanced across change groups as far as possible. At each history save:

- current mean/covariance and residual-consistency status;
- every base motion and its fastest admissible speed;
- torque and clearance margins;
- expected score for each selection rule;
- selection under the prior fixed order, B0 plan, rule, variance score, and decision score.

Within the same public context, report how often the chosen motion changes. Separate changes caused by public context, current admissibility, posterior mean, posterior covariance, and residual fallback.

Changing a fixture identity is not enough to count as evidence-dependent choice. Do not use a hidden group or family to choose an action.

### 11.3 Actual continuations from matched states

At those 24 histories, clone the complete evaluator state: integration state, q/dq, controller state, routing, persistent bias, AR state, and RNG state. This operation is evaluator-only.

Run up to four distinct candidate continuations:

- the fixed order's next eligible motion;
- the B0 plan's next eligible motion;
- the decision selector's motion;
- one additional diverse eligible candidate chosen by a fixed public-action rule.

Merge duplicates. For each candidate, use four fresh future-noise continuations with the same persistent state; the future stream changes across repeats, not the past. Use matched future streams across candidate branches where the sensor contract permits.

After the candidate, apply the same estimator and remaining fixed reference suffix, then verification. Record actual total remaining time, parameter errors, work prediction errors, accepted audit outcomes, and violations.

Publish both immediate parameter gains and eventual time-to-correct-release. A candidate that reduces parameter variance but does not help the job should be visible.

These per-hidden-case best branches are hindsight references. They are not available to deployed selectors and do not alone prove that an online selector can identify the best branch. The held-out whole-policy comparison provides that evidence.

### 11.4 Follow at least two genuine comparisons in detail

Select examples by a frozen rule after all results: largest time saving with correct audit, largest adaptive loss, and an example with identical schedules. Show exact measurements and action scores, not a schematic invented after the fact.

If S5 remains best nearly everywhere, print that result. Do not search for a replacement fixture or payload range until a different action wins.

---

## 12. Qualification and freezing

Run stages in this order:

1. **Audit parent:** verify code and source artifact hashes, separate plant/controller models, measurement identity, and actual torque semantics.
2. **Build public workstations and trajectories:** kinematic checks and source-only known-parameter tracking qualification.
3. **Fit shared estimator/predictor:** source data only; reserve source cases for uncertainty and prediction-margin calibration.
4. **Run source action maps and matched continuations:** explain actual mechanism and compute cost.
5. **Validate complete methods:** choose the fixed template and one shared uncertainty-inflation setting.
6. **Seal:** all manifests, settings, code, fallback rules, horizons, model parameters, and selected templates.
7. **Evaluate every test case for every main method and reference.**
8. **Audit, report and package.**

For shared uncertainty inflation, use a predeclared multiplier grid {1, 2, 4} on top of the fitted base covariance. Each choice is evaluated on all validation cases for all methods. Select the smallest multiplier passing source coverage checks and zero observed validation physical violations/false releases across the deployable methods. If none does, retain multiplier 4 as an explicitly labeled experimental setting, report the failure of qualification, and do not call the screen certified. Fix implementation errors before sealing; do not revise test requirements to obtain a winning method.

Keep physical tracking acceptance separate from engineering qualification. The existence of some non-completable work cases is a scientific outcome, not automatically an invalid simulator. Unexpected numerical instability or a broken public/private boundary is an implementation failure.

If source trajectories are all admissible and all jobs already clear after B0, record this before test launch and still complete the registered primary matrix within the budget. Do not endlessly tune until adaptive branching appears.

---

## 13. Preserve two separate comparisons

### Primary comparison: additional feedback after B0

Compare **Rechoose for Work Decisions** against **Plan Once After Initial Measurements**. Both get B0, the same estimate, the same possible motions, the same updates/gate/stop, and the same planning search budget where applicable. Only later recomputation of motion priority differs.

### Other comparisons

- Prior-Designed Fixed Order vs Plan Once: value of initial case-specific measurements for planning.
- Hand-Written Parameter Rule vs Rechoose for Work Decisions: value beyond a simple feedback rule.
- Rechoose for Work-Torque Variance vs Rechoose for Work Decisions: value of focusing on the actual work decision rather than only reducing uncertainty.
- Initial Test Only vs the other methods: value of collecting additional measurements at all.
- True-parameter reference: what the declared controller/job can accomplish when parameters are known.

Changing acceptance rules, controller gains, noise levels, or workload speed between methods is forbidden. Changing them between V1 and V2 must be recorded and is not attributed to adaptive selection.

---

## 14. Metrics, denominators, and the definition of an improvement

### 14.1 Results per run

Save at least:

- approved at normal speed / approved at half speed / unresolved / aborted;
- actual private audit pass count out of eight;
- false release: approved but one or more private work audits fail;
- all violations during **identification**, verification, and audit, separately;
- actual test-motion, travel, settling, verification, estimator, prediction, planning, and setting-update times;
- physical seconds and measured serial deployment seconds;
- parameter error and interval coverage, for evaluation only;
- path RMS/peak/final-hold error, saturation duration, peak requested/applied torque, minimum clearance;
- motion order, speed choices, repeat count, deferred/rejected patterns, fallback reasons;
- exact budget-exhaustion and release/stop reason.

The primary quality numerator is correctly approved normal-speed jobs; its denominator is every run, not just released jobs. Also report false release over all runs and over released runs. Half-speed success is separate, not silently equivalent to normal-speed throughput.

### 14.2 Timing without rewarding abstention

Use both:

1. Time comparisons restricted to paired runs where both methods achieve correct normal-speed release.
2. An all-case capped time: actual serial time for correctly approved normal-speed jobs; otherwise the common cap of 80 s physical plus the registered maximum online-compute allowance. Report the exact cap and do not interpret this as measured physical time.

Also plot correct normal-speed releases reached by physical time t. Runs unresolved, aborted, incorrectly released, or released only at half speed do not enter the normal-speed success curve. This makes "stop immediately without approving anything" visibly different from completing fast.

Report half-speed curves separately and raw resource distributions. No post-hoc weighted sum of bytes, CPU and robot motion is the main metric.

### 14.3 Statistical unit

Average each method's three noise seeds within each object-context case. Bootstrap payload families, then objects within family, keeping all contexts of an object together. Use 2,000 paired replicates. Show context-specific and family-specific directions as well as pooled means.

Primary descriptive success criterion:

- adaptive normal-speed correct-release rate is no more than 1 percentage point below the B0-plan comparator;
- adaptive false-release/violation rate is no more than 1 percentage point above it;
- adaptive saves physical test time, with a paired interval and effect in seconds;
- serial time is also reported so extra planning cannot be hidden.

Use one-sided 95% intervals for the two stated margins and a two-sided interval for time. With only 64 different objects, a zero error count does not certify a 1% population risk. If intervals do not establish a margin, report the measured counts and interval rather than selecting only successful cases.

Comparisons beyond the predeclared primary contrast are secondary. An action-sequence difference, by itself, is not a performance gain.

---

## 15. Mandatory separation of causes

The report must answer these questions in order:

1. How many objects already complete the specified job with old settings?
2. How many need a parameter update, slower operation, or a different controller/job?
3. Does the current model correctly predict the observed work errors and constraints?
4. Do measured values change which test patterns are admissible, at the same public context?
5. Among motions admitted by the **same** screen, do measured values change which one is selected?
6. Does that change improve actual time-to-correct-release against Plan Once After Initial Measurements?
7. Is any gain instead explained by early stopping, faster admissible execution, different parameter updates, or simply a different public workstation?
8. How much actual computation did the online selection require?
9. Does a simple hand-written feedback rule obtain the same gain?
10. Which remaining failures are measurement/model failures, selection failures, or work requirements the tested controller cannot meet?

Do not attribute gain solely to adaptivity unless the matched baseline comparison supports it. This is the central scientific separation, not an extra appendix disclaimer.

---

## 16. Engineering tests required before the main run

Implement and save these checks:

- **Parent preservation:** hashes of V1 and prior robot studies unchanged.
- **Actual payload effect:** at fixed configuration, changing the physical payload changes measured motion/torque.
- **Configuration-only effect:** updating the controller does not change the physical object.
- **No private leakage:** selection APIs have no plant access; swapping evaluator labels with the same public history changes no action.
- **Recording identity:** duplicate stored events do not create new measurements; independent repeats retain new identity.
- **Persistent bias:** bias and AR state survive block boundaries and return motions.
- **Time scaling:** geometric path is unchanged; velocity and acceleration scale correctly; duration and logging costs change.
- **No reset shortcut:** planned transitions begin from the public measured state estimate; actual plant state is continuous.
- **Frozen-order contract:** feed two histories with identical public context to a fixed method; ranking remains unchanged, while shared screening may differ and is logged separately.
- **Plan-once contract:** after B0, its priority list remains identical despite future posterior changes.
- **Equal wrappers:** estimator, update, screen, speed selection, watchdog and stop implementation are the same callable objects for all methods.
- **Prediction provenance:** hypothetical models contain sampled/public parameters, not actual test parameters.
- **Budget ledger:** initial hold, failed attempts, rejected/aborted physical actions, return travel, and verification are charged.
- **Action cache:** reuse requires identical complete initial state, controller settings, geometry, action, sensor/random state and code; reused observations cannot count as new independent evidence.
- **Branching:** cloned prefixes reproduce bit for bit before different actions are applied; hidden clone state stays evaluator-only.
- **Metric sanity:** normal-speed and half-speed success are different; final hold is checked over its full interval, not only the endpoint.
- **No evidence fabrication:** predicted rollouts are not added to the acquired measurement history.

---

## 17. Compute and storage budget

This round is CPU-first. Do not start GNN/RL training or request a GPU merely because previous runs used B200s.

Start with at most eight physical workers, one BLAS thread each. Profile 20 source blocks and public prediction calls; then raise workers only within the CPU/memory allocation actually authorized in the current environment. Historical SSH hosts and ports in V1 are not permission to use those machines now.

Maintain separate ledgers:

1. actual unknown-plant simulation seconds, including source, failures, verification, audits and evaluator branches;
2. hypothetical/public-model integration seconds;
3. CPU core-hours and wall time for estimation, screening and planning;
4. exact cache hits and reused artifacts.

Default hard ceilings:

| Resource | Cap |
|---|---:|
| Unknown-plant simulation | 2,000,000 simulated seconds |
| Hypothetical/public-model integration | 1,000,000 simulated seconds |
| Total CPU use | 160 core-hours |
| Final analysis ZIP | < 500 MB |

These are execution ceilings, not runtime predictions. Estimate totals from the source profile before dispatch. Use analytic calculations and exact caching where possible; do not silently reduce test cases, drop methods, skip failed runs, or increase budgets.

If the forecast exceeds a ceiling, save a stage-complete partial package and a concrete cost breakdown before requesting approval for expansion. Do not launch an unbounded set of hypothetical rollouts.

Large raw sensor and private-physics arrays may remain in the external run store. The ZIP must include complete per-case summaries, posterior/action scores, immutable file references, hashes, representative raw traces, code, and enough metadata to locate every raw artifact. A reference index is not the raw data itself; label it correctly.

---

## 18. Suggested implementation structure

Add the following new modules or equivalent clearly named components:

```text
payload_adaptive_tests_v2/
  README.md
  protocol.py
  cases.py
  workstations.py
  trajectories.py
  prediction.py
  motion_screen.py
  shared_decision.py
  fixed_plans.py
  policies.py
  branch_checks.py
  runner.py
  evaluate.py
  reporting.py
  packaging.py
  tests/
```

Reuse V1 through explicit imports or a versioned adapter. Avoid mutating parent module globals. The new run directory owns all modified configurations.

Provide these stage commands, or document equivalent real commands:

```bash
python -m payload_adaptive_tests_v2.runner --stage audit --run-root RUN_ROOT
python -m payload_adaptive_tests_v2.runner --stage qualify-source --run-root RUN_ROOT
python -m payload_adaptive_tests_v2.runner --stage fit --run-root RUN_ROOT
python -m payload_adaptive_tests_v2.runner --stage branch-checks --run-root RUN_ROOT
python -m payload_adaptive_tests_v2.runner --stage validate --run-root RUN_ROOT
python -m payload_adaptive_tests_v2.runner --stage seal --run-root RUN_ROOT
python -m payload_adaptive_tests_v2.runner --stage evaluate --run-root RUN_ROOT
python -m payload_adaptive_tests_v2.runner --stage report --run-root RUN_ROOT
python -m payload_adaptive_tests_v2.runner --stage package --run-root RUN_ROOT
```

These are interfaces to implement, not claims that the commands already exist. Every stage must be resumable with input hashes, exit codes and completion markers. Never mark an incomplete stage passed.

---

## 19. Required result files and figures

### Tables and traces

- `PARENT_AUDIT.md`
- `protocol_locked.json`
- `configs/workstations.json`, `work_requirements.json`, `test_library.json`
- public/private manifest hashes and split checks
- `source_prediction_validation.csv`
- `source_uncertainty_coverage.csv`
- `source_action_rankings.csv`
- `source_actual_branch_outcomes.csv`
- `validation_fixed_template_results.csv`
- `per_run_results.parquet`
- `per_step_actions.parquet`
- `candidate_scores.parquet`
- `work_reference_categories.csv`
- `paired_method_comparisons.csv`
- `identical_schedule_pairs.csv`
- `constraint_and_false_release_counts.csv`
- `time_breakdown.csv`
- `resources/ledger.jsonl`
- `FINDINGS_CN.md`
- `REPRODUCE.md`

### Figures

Use a consistent palette and show denominators.

1. The three public workstations, declared paths, and the test-motion library. Use actual simulator renders or accurately labeled geometry plots, not a generated imaginary robot scene.
2. Source action map: rows are actual post-B0 histories; columns are base motions. Show score, permitted speed and whether the shared screen admits the motion.
3. Correct normal-speed approvals vs physical commissioning time, all cases retained.
4. Correct normal-speed coverage vs all-case capped time; separate half-speed, unresolved and false releases.
5. Actual paired time difference: Rechoose for Work Decisions minus Plan Once After Initial Measurements, by context and payload family.
6. Predictive vs actual maximum tracking error, requested torque and minimum clearance.
7. Three real investigation timelines: adaptive saves time, adaptive loses or ties, and a common failure/unresolved case.
8. Change in action priorities after actual readings, separated from changes caused only by the common motion screen.

For paper-ready plots, provide the plotted CSV and a standalone TikZ/PGFPlots source per panel. Actual scene renders may accompany these as explicitly labeled simulation views. No figure may imply a causal path or a hardware experiment that was not executed.

### Final report wording

Use concrete sentences, for example:

> After the first tilt, the measured torques changed the estimated mass range from ... to .... Test D2 at normal speed then passed the same motion screen used by every method. The fixed order executed S5 next; the decision method executed D2. On these matched cases, both passed the eight audit jobs, and the decision method saved ... s, including planning time ....

Also show the opposite or tied cases. Explain whether the fixed comparator's common screen already provides most of the feedback benefit.

Do not write "the method improves robustness/efficiency" without stating what error, time or resource changed and on how many cases.

---

## 20. Evidence and source references

### Project sources reopened for this plan

**P1.** `Payload_Change_Commissioning_20261116T180000Z_GPT_Pro.zip`, especially:

- `payload_change_v1/code/payload_change_v1/design.py`: `build_design_bank`, `fixed_schedules`, `assess`, `choose_action`;
- `payload_change_v1/code/payload_change_v1/cases.py`: physically consistent payloads, range, group definitions, split generation;
- `payload_change_v1/code/payload_change_v1/trajectories.py`: original reference, six static/four dynamic patterns, workload limits;
- `payload_change_v1/code/payload_change_v1/estimator.py`: the 17-variable update and approximate physical projection;
- `payload_change_v1/code/payload_change_v1/protocol.py`: resource accounting and sealed requirements;
- `payload_change_v1/code/payload_change_v1/documents/CONTROLLER_DECISION.md`: actual controller, sensors, separate models, L0 qualification and cloning.

**P2.** `payload_change_review/verification/recomputed_main.csv`: equal results of the three V1 multi-test methods, and the keep-settings and single-task references.

### External primary sources checked for motivation

**E1.** Bohao Zhang, Zichang Zhou, Ram Vasudevan. *Provably-Safe, Online System Identification*. Robotics: Science and Systems, 2025. DOI: `10.15607/RSS.2025.XXI.121`.

- https://roboticsproceedings.org/rss21/p121.html
- https://arxiv.org/html/2504.21486v1
- https://roahmlab.github.io/OnlineSafeSysID/

**E2.** Matthew Cooper, James McGree, Timothy L. Molloy, Jason J. Ford. *Bayesian Experimental Design With Application to Dynamical Vehicle Models*. IEEE Transactions on Robotics 37(5):1844-1851, 2021. DOI: `10.1109/TRO.2021.3063977`.

- https://research.monash.edu/en/publications/bayesian-experimental-design-with-application-to-dynamical-vehicl/

**E3.** Universal Robots. *Set Payload*, PolyScope X manual, four-position payload/CoG wizard.

- https://www.universal-robots.com/manuals/EN/HTML/SW10_14/Content/prod-usr-man/complianceUR3e/PolyX_SW/psx_payload_set.htm

**E4.** Franka Robotics. `franka::Robot::setLoad` API: configured payload mass, center of mass and inertia.

- https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Robot.html

All new context geometry, precision requirements, budgets and numerical experiment choices above are proposed specifications for this V2 experiment. They are not measurements from those external papers or declarations of manufacturer-certified operation.

---

## 21. What to return to the project owner

Provide the analysis ZIP and a brief summary in Chinese answering:

1. Did actual measurements change the next useful test within the same public workstation?
2. Did recomputing the order after later measurements beat planning once after B0?
3. How many seconds did it save, and what happened to work correctness and violations?
4. Did the simple rule or fixed plan already obtain the same result?
5. Which concrete measurement changed which concrete choice?
6. If no extra gain appeared, was the reason early completion, a universally useful fixed motion, a screen that admitted everything, unreliable prediction, or poor action scoring?

Finish with the measured result, not with a promise that a harder task would make the proposed method win.
