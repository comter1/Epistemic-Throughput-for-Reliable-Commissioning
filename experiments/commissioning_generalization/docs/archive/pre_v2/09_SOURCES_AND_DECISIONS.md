# 09. Sources, provenance and implementation decisions

## Repository inspected

Local uploaded snapshot:
`Epistemic-Throughput-for-Reliable-Commissioning-master.zip`.
It contains a finite discrete benchmark, plotting scripts and archived result summaries.
It does not contain a trained GNN or the new MuJoCo system. Original toy model has five
primitive measurements, up to30 combination actions, costed calibration, exact posterior,
greedy and restricted DP. Preserve it as a regression/reference test, not a target-system
probability model. Hashes are in the handoff's repository_snapshot.json.

The previous PDF supplied the overall information/budget model, candidate graph-based
value learning, physical simulator plan, and structural holdouts. CGX-1.0 makes a main
experiment operational, including variable-size fault-location targets, fully specified
split counts and explicit data/clock boundaries. This package does not reproduce any
unreleased paper text or require the PDF to execute.

## Public primary documentation (checked 2026-11-13)

1. OpenAI, Codex scoped project instructions:
   https://developers.openai.com/codex/guides/agents-md
   Usage here: nested AGENTS.md under the new experiment; do not override existing root instructions.
2. OpenAI, Iterating Development Workflows with Codex:
   https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex
   Usage here: explicit scope, implementation phases and saved execution evidence.
3. MuJoCo Python documentation:
   https://mujoco.readthedocs.io/en/stable/python.html
   Usage here: native bindings, direct stepping without interactive viewer.
4. MuJoCo simulation and API reference:
   https://mujoco.readthedocs.io/en/stable/programming/simulation.html
   https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
   Usage here: actual step/control loop, separation of simulation time and wall time.
5. Google DeepMind Menagerie Panda:
   https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda
   https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/README.md
   Usage here: pinned asset, its specified actuators/joint limits and license.
6. PyTorch Blackwell support / installation matrices:
   https://pytorch.org/blog/pytorch-2-7/
   https://pytorch.org/get-started/previous-versions/
   Usage here: compatible B200 build and driver. 2.7 added support, but is not prescribed
   as the latest release. Inspect the host and freeze a tested compatible build.
7. Deep Adaptive Design (Foster et al., ICML2021):
   https://proceedings.mlr.press/v139/foster21a.html
   Relevant to learning an acquisition/design policy, not a claim of inventing that idea.
8. Efficient Differential Diagnosis using Cost-aware Active Testing (2024):
   https://papers.phmsociety.org/index.php/ijphm/article/download/3849/2503
   The primary simulator's learned next-loss comparator is an adaptation of its
   entropy/cost selection idea, not a reproduction of its system-specific causal model.

Version-dependent installation details must be checked at implementation time and locked.
Do not put literal “latest” package versions in the final reproduction environment.
Do not claim a source guarantees the proposed empirical performance or runtime.

## Deliberate design decisions and clarified earlier ambiguity

- Main contribution is held-out-connection acquisition behavior; GNN/RL/IG alone are not novelty.
- Primary target includes location because multiple queues/caches make type alone insufficient.
- Diagnosis can use graph information. Equal shared q, not an intentionally weak q, isolates selection.
- Transformer also gets connections; withholding topology is a separate ablation.
- Decisions, executions and calibrations are separate counts (default6,4,2).
- Fixed plan means fixed after X0, not “always obtain one full bundle”. Its learned implementation
  is not labeled globally optimal.
- Calibration is measurement, not setting clock error to zero. Two-way timestamp exchanges
  retain asymmetry/noise. Perfect noise cancellation remains only an old toy condition.
- Source/validation/test topology identities and scope-specific overlaps are explicit.
- Smaller input schema and known types are allowed; target probability tables/labels are not.
- Exact ET and optimal frontiers are reported only where genuinely computed in finite fixtures.
- Hardware supports this model size; actual experiment time is a measured planning quantity.
- The branch is created by Codex in the user's checkout. This handoff has not changed their Git state.
