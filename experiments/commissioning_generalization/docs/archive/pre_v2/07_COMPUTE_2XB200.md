# 07. Two-B200 resource plan

## Hardware use

Two B200 GPUs are available per the user. Use independent jobs, one/GPU. This experiment
has small graph and temporal networks, not an LLM; no multi-GPU model is required.
Actual memory and GPU partitioning come from nvidia-smi and torch; do not assume an
unpartitioned advertised memory capacity. Test BF16 forward/backward and standard scatter
operations. Keep score calculations and log losses FP32. No FP8 requirement.

CPU MuJoCo and Python components are the authoritative execution engine in v1. More GPU
memory does not accelerate CPU queues, recorders, or simulation by itself. MJX/Warp is an
optional future implementation, not an automatic substitute: it must preserve actual
fault, clock, logging and timing semantics before being compared.

## Workload, not a promised completion time

Source cases=128*256=32,768. Eight trajectories/case produce262,144 trajectories.
Typically several decisions per trajectory: 1–2M transitions is a collection target,
with hard total source physical-execution cap2M. Actual count may be smaller; report it.
Five policy families*five seeds*200k updates=5,000,000 policy updates.
Three crossfit+one final+one independent diagnostic readers at100k cap=500,000 reader updates.
No automatic extra large hyperparameter sweep; checkpoints and listed scalar thresholds
are selected on validation. If changing learning rate is needed for a numerical failure,
apply the amended comparable setting to all affected methods and log it.

For a planning estimate ONLY, 5.5M updates at0.05/0.10/0.20 s per update are
76.4/152.8/305.6 GPU-hours total, or38.2/76.4/152.8 ideal wall-hours on two GPUs.
Reader and policy updates need not take equal time; actual profiling supersedes this
rough range. It excludes simulation, I/O, evaluation, retries and auditing. The initial
soft cap320 GPU-hours triggers a report, not hidden pruning of seeds or methods.

Main test cases=5*20*128=12,800. At five byte budgets and25 trained policy runs,
there are up to1.6M learned-policy case evaluations, with up to4 executions each in
main caps (byte0 often needs none). Rules/references, validation, source collection,
round sensitivity and repair audits add workload. This can cost MORE than training.
Reuse identical nonlearned-policy runs across seeds and reader predictions for identical
transcripts, but do not fabricate runtime savings by treating cached execution as live.

Repair subset is5 suites*4configs*64cases=1280cases. Four methods, three learned seeds
(nonlearned reuse), two public runs and up to20audit runs per accepted case. Report actual
counts; not every case is accepted. Evaluation/audit may dominate CPU demand.

## CPU and storage

Recommend128GB system RAM, >=16 actually allocated cores, and initially1TB free SSD,
prefer2TB if keeping extensive raw logs. These are planning requirements, not a claim
about the user's host. Use hardware_probe.py to record the actual allocation.
A full-rate run can be several MB: e.g.64ports*1000records*128B is8.192MB before external
records. One million such full runs would exceed8TB. Therefore the protocol must use
costed selective recording, compressed deduplicated transcripts, and explicit raw retention.
Do not promise the whole corpus fits into1TB without measuring the emitted format.

Before the formal queue, time1000 technical executions and1000 representative updates,
including recorder and encoder. Estimate remaining work from measured percentiles and
actual manifest counts; save reports/resource_forecast.json. This is capacity planning,
not an empirical pilot used to select favorable hypotheses. If capacity is insufficient,
change only parallelism/retention or ask for an explicit protocol amendment; never shrink
held-out tests without disclosure.

For CPU planning, each1M simulator executions at1/5/10 CPU-seconds each cost roughly
278/1389/2778 core-hours. Divide by effective parallel workers, not by installed GPUs.
Use as-fast-as-possible sim-time scheduling; real-time sleep would waste this allocation.

## Measured timing boundary

Report source-data generation, reader training, Q training, target-system acquisition,
policy selection, diagnostic inference, public verification and external auditing separately.
Warmup/JIT/asset load excluded from steady-state latency must also be reported separately.
Do not compare GPU-batched throughput against one-case CPU latency as if same resource.
Reference model-based methods get the same per-decision online time cap; their model
construction cost is additional, not silently free.

## Preflight and software lock

Record GPU UUID, name, memory, current utilization, driver, compute capability, torch
build/runtime CUDA, Python, MuJoCo version, package lock, CPU affinity, RAM/free disk,
Menagerie SHA and OS. Do not upload keys/tokens or full user environment variables.
Use official installation matrices. No automatic host driver upgrade, privileged Docker
configuration or global pip changes. A compatible preinstalled environment can be reused
in an isolated clone/venv, then frozen.
