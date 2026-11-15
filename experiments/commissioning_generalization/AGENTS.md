# Scoped instructions: commissioning generalization main experiment

This directory implements MAIN EXPERIMENT V2 (CGX-2.0). Read `docs/MAIN_EXPERIMENT_V2.md` and
`configs/protocol.json` before coding. Read the other docs for interface contracts.

- Edit this directory only. Legacy finite experiments and results are read-only.
- Implement a real headless MuJoCo execution path. Do not substitute synthetic
  diagnostic numbers for physical sensor/cache/controller/queue/actuator behavior.
- Graph counts and candidate counts vary. Score nodes and actions with shared heads;
  never flatten all nodes into a fixed-length feature vector or use fixed action logits.
- Test systems receive frozen weights. Only acquired history/belief changes online.
- Public topology, nominal parameters, legal actions and declared costs are allowed.
  Fault metadata, true offsets, evaluator state and unacquired records are forbidden.
- Preserve message equality but never expose monotonically numbered IDs, simulator
  tick IDs or true global event timestamps through the agent boundary.
- No target-system label training, probability-model fitting, or hidden simulator queries.
- q_phi is the common diagnosis model. Q_psi is the acquisition value model.
  Reward uses true labels only in training; the runtime policy never sees labels/rewards.
- Cost and decision/execution horizon are separate. STOP is feasible; timeouts and
  abstentions remain in evaluation. Calibration estimates an offset, never repairs it.
- Main runs do not use exact common-noise cancellation; independent residual error remains.
- Same information/action/budget access for comparators. Transformer sees the edge list too.
- Use standard PyTorch operations for the initial GNN to avoid unnecessary compiled
  scatter-extension dependencies. Every trained run records the exact environment.
- No image input, no trained robot controller, no paid model API calls.
- Three B200s (two local, one remote) run independent jobs; no automatic DDP, driver updates or Docker daemon changes.
- Configurations and test splits are sealed before training; results cannot select test cases.
- Technical smoke tests are not substitute research experiments. Proceed to the main protocol.
- No claims of exact ET or Bayes optimality from learned probabilities or approximate search.
- Follow `docs/08_ACCEPTANCE_TESTS.md`; preserve actual evidence of tests.
- Keep generated trajectories/checkpoints outside Git in CGX_DATA_ROOT.
