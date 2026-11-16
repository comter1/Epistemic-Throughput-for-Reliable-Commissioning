# Portable evidence indexes

`evidence_export.py` is a delivery/integrity tool. It never calls a reader or
selector, fits parameters, assigns measurement identities, or acquires data.
Run it on the remote compute hosts with the normal experiment `PYTHONPATH=.:src`.

```bash
python -m structured_evidence_v1.evidence_export old_histories \
  --source-root /work/Users/leiyo/cgx_data_v2 --run-dir RUN --workers 12
python -m structured_evidence_v1.evidence_export new_main \
  --source-root /work/Users/leiyo/cgx_data_v2 --run-dir RUN --workers 12
```

The first command requires completed old-history replay. The second requires the
complete official 147,200-cell main matrix and its frozen model manifest. The
commands write separate `evidence_export/<cohort>/` directories. A completed export
is preserved; rerunning verifies its output hashes instead of overwriting it.

Each directory contains:

- `histories.jsonl.gz`: packed public histories, acquisition identity envelopes,
  canonical history/provenance addresses, source paths and context keys. The old
  cohort covers every replay job, including inline diagnostic prefixes that were
  never independent files in the original store. IDs are copied from the actual
  saved structured prediction shards. The new cohort covers every normal final
  history, the initial prefix and each recorded decision prefix, including STOP
  or a failed acquisition with no new observation. Prefixes project the saved
  observation chain and recorded resources; they do not invent event IDs.
- `traces.jsonl.gz`: for the new cohort, one compact row per official main cell
  containing its full recorded result, decision trace and original result hash.
  Thus all cells, including recorded outer failures, remain available without
  adding 147,200 separate files to the analysis ZIP. This file is empty for the
  old cohort, whose context/prediction tables are already delivered separately.
- `blobs.jsonl.gz`: dependency closure for every graph, task and observation,
  including store search order, resolved external path, content-address key,
  compressed size and SHA256. Repeated references share a locator entry.
- `metadata.jsonl.gz`: graph/task JSON values, with verified canonical addresses,
  so topology, ports, units, clock groups and nominal task metadata are portable.
- `failures.jsonl.gz`: every failed context/dependency verification. Unexpected
  missing references make completion fail. Explicit outer rollout failures with
  no saved history are retained in `traces.jsonl.gz` and counted separately.
- `complete.json`: input/output hashes, coverage and failure counts, verification
  scope, implementation hash, elapsed time and worker allocation.

The JSONL files are gzip streams, one JSON object per line. All final probability
vectors remain in the separately delivered main and replay prediction Parquet
tables. Join old histories on `job_id`; join new histories/traces on `cell_id` or
the five fields suite, method, policy_seed, execution_cap and case_id. A null
policy seed denotes Full Recording.

Large raw observation values remain in the immutable external stores. Their
compressed bytes are hashed without reparsing all record JSON; this verifies
their current byte identity and availability, not their historical correctness.
Canonical JSON addresses are checked for packed histories, provenance, graphs and
tasks. Recomputing a raw observation's canonical JSON hash is an additional check
available to an auditor using its locator. The analysis ZIP is consequently a
portable index and analysis package, not a self-contained raw-data archive.

Recorded predictions are copied exactly. Plan-Once may record the previous
posterior at a prefix with `diagnosis_updated=false`; that is not a fresh diagnosis.
Conversely, `diagnosis_updated=true` records an attempted diagnostic call, which
can fail or time out and retain the previous posterior. Interpret it together
with the original trace stop reason and row error strings, which are also copied.
The frozen rollout did not save a full initial posterior vector in new-main rows;
the export marks it unavailable instead of fabricating it. It does retain the
initial acquired evidence and every officially saved final and trace prediction.
Identity fields remain equality/grouping metadata, not numerical model inputs.
