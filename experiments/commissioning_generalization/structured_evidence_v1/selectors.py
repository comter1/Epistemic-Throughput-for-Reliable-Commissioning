"""Unchanged selector networks with read-only access to the original replay.

This module changes neither policy features nor the learning objective. It
separates immutable source transcripts/features from new structured-reader
predictions, rewards, checkpoints and evaluation writes.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import resource
import sqlite3
from pathlib import Path

from cgx.models.features import FEATURE_VERSION
from cgx.training.feature_cache import read_features
from cgx.training.replay import Replay, canonical_hash, hash_file


class ReadOnlyReplay(Replay):
    """Use the already prepared original SQLite/mmap data without touching it."""

    def __init__(self, source_root, run_root, split="train"):
        from .storage import OverlayStore
        self.source_root, self.root = Path(source_root), Path(run_root)
        self.split = split
        self.store = OverlayStore(source_root, run_root)
        self.path = self.source_root / "training" / f"replay_{split}.sqlite"
        if not self.path.is_file():
            raise FileNotFoundError(f"Existing prepared source replay required: {self.path}")
        self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        self.source_hash = self.db.execute("SELECT value FROM meta WHERE key='source_hash'").fetchone()[0]
        self.plan_counts = dict(self.db.execute("SELECT case_id,COUNT(*) FROM plan_support GROUP BY case_id"))
        self.case_rows = self.db.execute("SELECT case_id,n,fold,configuration FROM cases ORDER BY case_id").fetchall()
        feature_key = canonical_hash([FEATURE_VERSION, hash_file(Path(__file__).parents[1] / "src/cgx/models/features.py")])
        self.feature_directory = self.root / "encoded" / feature_key
        self.original_feature_directory = self.source_root / "encoded" / feature_key
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = 16384 if hard == resource.RLIM_INFINITY else min(16384, hard)
        if soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
            soft = desired
        self.cache = lru_cache(maxsize=min(1024, max(16, soft // 8)))(self._features)
        # The source has 185,018 train histories. Keep their small immutable
        # graph-reference/budget metadata after first use; no RNG or math changes.
        self.state_cache = lru_cache(maxsize=262144)(self._state)
        self.graph_cache = lru_cache(maxsize=512)(lambda ref: self.store._get("graphs", ref))
        self._prefetch_pool = None
        self.eligible = {fold: [r for r in self.case_rows if r[2] != fold] for fold in (None, 0, 1, 2)}

    def _features(self, ref):
        original = self.original_feature_directory / ref[:2] / (ref + ".cgxf")
        if original.is_file():
            return read_features(original)
        # A missing old cache is rebuilt exclusively inside the new namespace.
        return super()._features(ref)


def source_history_provenance(rows):
    """Recover acquisition identity from executed transition chains, not values.

    Initial executions are shared within a case collection/coverage job. Later
    IDs identify their actual trajectory and acquired step. Opaque identity
    strings are only equality keys, never numerical model inputs.
    """
    trajectories = defaultdict(list)
    for row in rows:
        trajectories[row["trajectory_id"]].append(row)
    output = {}
    def retain(ref, provenance):
        previous=output.setdefault(ref,provenance)
        def partition(ids):
            lookup={}
            return [lookup.setdefault(value,len(lookup)) for value in ids]
        if partition(previous['observation_ids'])!=partition(provenance['observation_ids']):
            raise ValueError('One source history has conflicting acquisition-identity equality patterns')
        # An opaque renaming with the same equality pattern carries the same
        # evidence. Preserve the first exact chain identity for its audit.
        return previous
    for tid, trace in sorted(trajectories.items()):
        trace.sort(key=lambda r: r["step"])
        first = trace[0]
        initial = canonical_hash(["structured-source-initial-v1", first["parent_case_id"], first.get("coverage_branch")])
        ids = [initial]
        for row in trace:
            before = dict(observation_ids=list(ids), identity_origin="existing_source_transition_chain_v1")
            retain(row['history_hash'],before)
            if row["action"]["kind"] != "stop":
                ids = ids + [canonical_hash(["structured-source-acquired-v1", tid, row["step"]])]
            after = dict(observation_ids=list(ids), identity_origin="existing_source_transition_chain_v1")
            retain(row['next_history_hash'],after)
    return output


def model_spec(model):
    if model is None:
        return None
    return {"method": model.method, "width": model.width,
            "weights": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def construct_policy(spec):
    if spec is None:
        return None
    from cgx.models import PolicyModel
    from cgx.training.objectives import frozen
    model = PolicyModel(spec["method"], spec["width"])
    model.load_state_dict(spec["weights"])
    return frozen(model)


def reader_fingerprint(reader):
    """Model parameter identity; mutable inference caches are excluded."""
    if hasattr(reader, "parameter_fingerprint"):
        return reader.parameter_fingerprint()
    if hasattr(reader, "parameters_dict"):
        return canonical_hash(reader.parameters_dict)
    return reader.checkpoint_hash


class ImmutableScoreLookup:
    """Decode the sealed OOF labels once; repeated sampling uses exact values.

    Policy features consume just target keys and log2 probabilities. All audit
    envelopes remain in the immutable on-disk database; no new numerical dtype,
    rounding, probability calculation or sampling occurs during this preload.
    """
    def __init__(self,db):
        self.predictions={}
        for ref,fold,data in db.execute('SELECT history,fold,data FROM predictions'):
            prediction=json.loads(data)
            self.predictions[ref,fold]={name:prediction[name] for name in
                ('target_keys','log_probabilities_base2')}
        self.scores={key:(reward,next_nll) for key,reward,next_nll in
            db.execute('SELECT id,reward,next_nll FROM scores')}

    def prediction(self,ref,fold):
        try:return self.predictions[ref,fold]
        except KeyError as exc:raise RuntimeError(f'Missing out-of-fold prediction {ref}') from exc

    def score(self,key):
        try:return self.scores[key]
        except KeyError as exc:raise RuntimeError('Unscored replay transition') from exc


def consume_original_reader_initialization(source_root):
    """Match the original seed-to-policy initialization path exactly.

    Original fit_policy constructed its frozen DiagnosticReader after _seed,
    consuming Torch CPU RNG before constructing PolicyModel. The structured
    reader has no Torch parameters. Construct and discard only that original
    architecture here; its learned weights never enter the new diagnoser.
    """
    import torch
    from cgx.models import DiagnosticReader
    directory=Path(source_root)/'models/readers/final'
    marker=json.loads((directory/'complete.json').read_text())
    checkpoint=directory/marker['selected_checkpoint']
    if hash_file(checkpoint)!=marker['selected_checkpoint_hash']:
        raise RuntimeError('Original reader constructor source checkpoint changed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    spec=dict(architecture=payload['architecture'],width=payload['width'])
    discarded=DiagnosticReader(**spec)
    del discarded,payload
    return dict(**spec,source_checkpoint_sha256=marker['selected_checkpoint_hash'],
        purpose='match original CPU RNG consumption before fresh policy initialization',
        original_reader_weights_used=False)
