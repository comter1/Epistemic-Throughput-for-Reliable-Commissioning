# CGX-1.0-P48-1: staged delivery authorized before formal training

On 2026-11-13 the user requested a result within two days and explicitly selected:
all training/test systems, all five learned methods, primary budget, one seed,
and a fixed shorter training schedule first; remaining seeds, budgets and audits
follow. This supersedes the original order of delivery, not the original dataset.
No formal model had trained and no held-out outcome had been evaluated when this
amendment was made. The original numerical protocol file remains unchanged.

The first report uses seed11, 10,000 updates for each of three cross-fit readers,
the final reader and the independent reader, and 20,000 updates for each of the
five policies. Both neural architectures have the same update/batch/validation
budgets. Checkpoints and complete 3,072-case policy validation occur every5,000
updates. It retains all128 source configurations/32,768 cases, all24 validation
configurations/3,072 cases, and all100 held-out configurations/12,800 cases.
Eight designated source trajectories per case and coverage remain unchanged.

The first frozen evaluation covers byte fraction0.25, execution cap4, calibration
cap2, all five suites, all five learned methods, the four original nonlearned
controls, and cost-aware adaptation with validation-selected beta. There are
128,000 case/method cells. Independent-reader checks and configuration/case paired
bootstrap are included. Policy-seed variance, other budget points, horizon
sensitivity and downstream repair/audit are explicitly outstanding at this stage.
This is a single-seed shorter-trained primary result, not completion of CGX-1.0
or evidence of robustness across training seeds. No GNN advantage is assumed.

The second profile retains the original five seeds, 100k/200k update caps, five
budgets and full repair/audit requirements. It is sealed before the first profile
looks at test outcomes. Its fixed settings cannot be adjusted using the first
report. It follows automatically after the first report/export. Both profiles
have separate model, evaluation, result and protocol-seal directories.

The shared source corpus continues under the original archived, hash-verified
software and configuration. Existing collected cases are retained. A second CPU
collection process generates validation concurrently. The new profiles link only
the common immutable manifests, assets, public records and private case/transition
corpus; trained model/result directories are separate. Training readers open only
the train/validation private split files. Test access remains behind model freeze.

Runtime improvements use independent CPU main-thread rollout workers, with one
CPU thread per worker and unchanged per-call deadlines. Every timing is labeled
CPU inference/parallel process execution, not represented as B200 inference time.
GPU training continues as one independent task per GPU. A spawn process context
avoids inheriting a trainer's CUDA/optimizer state. Complete matrix inventories,
timeouts, failed acquisitions, abstentions, resource caps and evidence hashes stay
mandatory. Reader-validation and OOF inference batch existing encoded features;
training prefetch only decodes features concurrently and does not reorder RNG draws.

The initial target is 2026-11-15 21:00 UTC, approximately48 hours after the request.
It is a planning target, not an assertion that unexecuted jobs will finish on time.
Actual commands, failures, progress and the two completion scopes are recorded in
`CGX_DATA_ROOT/staged_run/status.json`. A late/incomplete report remains labeled
incomplete. The first and full outputs are under `runs/primary_48h` and
`runs/full_protocol`. Engineering requalification precedes their source seals.

The user additionally offered one B200 via `leiyo@ssh.cloud.sdu.dk:2127`.
At the initial attempt SSH authentication was unavailable. A task-specific public
key was provided for the user to install. Remote capacity is counted only after
successful login, hardware/occupancy/environment checks and a real GPU operation.
