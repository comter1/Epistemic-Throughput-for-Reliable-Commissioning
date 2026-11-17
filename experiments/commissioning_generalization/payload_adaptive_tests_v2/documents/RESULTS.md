# Completed run: 20261117T212456Z

The complete held-out matrix contains 3456 method runs (576 per method), 576
reference groups, 64 objects, three public workstations and three noise seeds.
All six methods correctly release all 576 normal-speed jobs, with eight passing
private audit jobs per release and no false releases. Every method stops after
8 s of B0 and performs the same 10 s verification. Replanning after later readings
therefore saves **0 physical seconds** against planning once after B0.

The old configured settings pass 435/576 standalone reference jobs; true parameters
pass all 576. All methods update parameters in 220/576 commissioning runs. Each
method has 144 B0 trajectories that do not meet the final job's stringent accuracy
criteria, reported separately from the zero observed contact/joint/torque constraint
violations. The robot is not approved until the common predictor and actual job
verification pass.

The 24 preselected source histories also already permit release after B0. Forced
continuation diagnostics contain 196 real matched branches. The fixed and plan-once
selectors choose S5 in all 24; the decision selector chooses S5 in 23 and S6 in one.
This internal choice difference is not an observed commissioning-time advantage.
Higher-precision 256/32 source checks are completed offline with paired deterministic
samples, without changing any frozen method or held-out result.

Validation includes every multiplier {1,2,4} and every fixed template: 2304 runs.
The selected covariance multiplier is 2 and the fixed template is 0. The workflow
preserves engineering attempts, a preseal timing correction and known unsaved raw
failed tails. The raw run store and analysis archive contain the full evidence and
limitations; no scientific claim is based solely on smoke tests or predicted scores.

Artifacts: /work/Users/leiyo/cgx_data_v2/payload_adaptive_tests_v2/20261117T212456Z
Start with FINDINGS_CN.md, paired_method_comparisons.csv,
constraint_and_false_release_counts.csv and work_reference_categories.csv.
