# V2 compute: three B200s

The historical filename is retained for links; the active allocation is THREE
B200s, one independent job per GPU. Two are local and one is available over SSH
at leiyo@ssh.cloud.sdu.dk:2127. Both hosts see the same repository, environment
and external data root. The remote hardware report records an actually executed
BF16 forward/backward and graph aggregation check, GPU UUID and shared-file hash.

Set CGX_REMOTE_WORKERS_FILE to the V2 reports/remote_workers.json registry. The
queue requires all three registered slots and acquires a per-UUID lock. It dispatches
five reader jobs and fifteen method/seed policy jobs with exact input hashes.
Remote trainers run detached, with heartbeat, logs, exit status and GPU memory
recorded in shared storage. The queue verifies checkpoint bytes and early-stop
validation history before marking a job passed. It never interrupts unrelated jobs.

CPU process pools evaluate independent MuJoCo cases. Default16 workers per active
job and one BLAS thread per worker avoid oversubscribing the96 local and48 remote
CPU allocations. Mapping immutable feature files requires more than the SSH default
1024 descriptors; each Replay raises its own soft descriptor limit within the OS
hard limit, and bounds its cache if that is unavailable. This is a process setting,
not a global system change. All precision, sample counts and scientific caps remain
registered. No DDP, model enlargement, driver updates or paid APIs are needed.

Measured throughput and all forecast assumptions are in reports/resource_forecast.json
under the external root. Update the ETA from actual command/job intervals. The320
GPU-hour soft threshold is reporting-only and cannot prune seeds or cases. Optional
appendix jobs cannot block the main result.
