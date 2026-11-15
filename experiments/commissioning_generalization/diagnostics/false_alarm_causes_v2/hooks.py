"""Exact frozen-reader P/C/E substitutions; no changes to original forward code.

Only public histories enter this module. Labels and evaluation outcomes belong in
the caller. Cached sequence embeddings retain observation/port acquisition IDs.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import copy
import pickle
import time

import numpy as np
import torch

from cgx.models.features import FEATURE_VERSION, tensorize_history, collate_features
from cgx.data.store import FIELDS
from cgx.provenance import hash_json

HOOK_VERSION = 'cgx.fa.v2.pce.1'
COUNT_COLUMNS = (23, 24, 25, 31)
EDGE_FIELDS = ('edge_index', 'edge_types', 'edge_features')
IGNORED_BATCH_FIELDS = ('resources', 'features', 'events', 'lengths', 'sequence_owners',
                        'sequence_batch', 'sequence_valid')


def sequence_metadata(history, provenance=None):
    """Order matches tensorize_history; equal content does not imply same acquisition."""
    if provenance is None:
        provenance = [f'acquisition:{i}' for i in range(len(history['observations']))]
    if len(provenance) != len(history['observations']):
        raise ValueError('One acquisition identity is required per observation')
    out = []
    for oi, (obs, acquisition) in enumerate(zip(history['observations'], provenance)):
        grouped = defaultdict(list)
        for record in obs.get('records', ()):
            grouped[record['port']].append(record)
        for port, records in grouped.items():
            out.append(dict(port=port, observation_index=oi, acquisition_identity=acquisition,
                            record_count=len(records), missing=any(r['missing'] for r in records)))
    return out


@dataclass
class ReaderState:
    identity: str
    graph_identity: str
    batch: dict
    local: torch.Tensor
    external: torch.Tensor
    sequence_vectors: torch.Tensor
    sequence_metadata: list
    node_lookup: dict
    clock_lookup: dict
    targets: list
    prior: list
    prediction: dict
    timing: dict
    provenance: list

    def edge_map(self):
        inverse = {i: ('node', name) for name, i in self.node_lookup.items()}
        inverse.update({i: ('clock', name) for name, i in self.clock_lookup.items()})
        result = {}
        for index, (a, b) in enumerate(self.batch['edge_index'].T.tolist()):
            key = (*inverse[a], *inverse[b], int(self.batch['edge_types'][index]))
            if key in result:
                raise ValueError(f'Nonunique endpoint/relation edge key: {key}')
            result[key] = self.batch['edge_features'][index].tolist()
        return result


def prediction_error(actual, expected, tolerance=1e-6):
    if actual['target_keys'] != expected['target_keys']:
        raise AssertionError('Target identities changed')
    a, e = (np.exp2(x['log_probabilities_base2']) for x in (actual, expected))
    error = float(np.max(np.abs(a-e)))
    same_argmax = int(a.argmax()) == int(e.argmax())
    if error > tolerance or not same_argmax:
        raise AssertionError(f'Probability reconstruction: max_abs={error}, same_argmax={same_argmax}')
    return dict(max_abs_probability_error=error, argmax_identical=same_argmax)


class FrozenReaderHooks:
    def __init__(self, model, ledger, cache_dir=None, detail='reader_A'):
        self.model, self.ledger = model, ledger
        self.detail = detail
        self.device = next(model.parameters()).device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.checkpoint_hash = getattr(model, 'checkpoint_hash', 'test-unpersisted-model')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Frozen eval model required')

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def prepare(self, history, provenance=None, features=None, identity=None, prepared_metadata=None):
        """One native model forward captures exact GRU outputs and native pools.

        A caller passing features MUST already have charged their construction.
        Optional identity is a verified public history/content identity, not a label.
        """
        provenance = provenance or [f'acquisition:{i}' for i in range(len(history['observations']))]
        if set(history) != FIELDS:
            raise ValueError('Reader accepts only the original public-history whitelist; no earlier q or truth')
        key = hash_json([identity or history, provenance, self.checkpoint_hash, FEATURE_VERSION,
                         HOOK_VERSION, str(self.device), 'fp32'])
        path = self.cache_dir / (key + '.pkl') if self.cache_dir else None
        if path and path.exists():
            self.ledger.reserve('cache_hit', detail='reader_prepare')
            with path.open('rb') as handle:
                return pickle.load(handle)
        t0 = time.perf_counter()
        if features is None:
            self.ledger.reserve('feature_build', detail=self.detail)
            features = tensorize_history(history)
        metadata = sequence_metadata(history, provenance) if prepared_metadata is None else prepared_metadata
        if len(metadata) != len(features.sequences):
            raise AssertionError('Sequence metadata differs from exact feature ordering')
        if np.any(features.nodes[:, 26:29]):
            raise AssertionError('Actual reader history unexpectedly contains planned actions or previous q')
        batch = collate_features([features], self.device)
        self._sync(); preprocess = time.perf_counter()-t0
        captured = {}
        def pools(module, args, output):
            captured['local'], captured['external'] = [x.detach().cpu().clone() for x in output]
        def sequence(module, args, output):
            captured['sequence_vectors'] = output[1][-1].detach().cpu().clone()
        handles = [self.model.backbone.evidence.register_forward_hook(pools),
                   self.model.backbone.evidence.gru.register_forward_hook(sequence)]
        self.ledger.reserve('model_forward', detail=self.detail+'_native')
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                lp = self.model(batch)[0, :len(features.targets)].cpu().tolist()
        finally:
            for handle in handles:
                handle.remove()
        self._sync(); elapsed = time.perf_counter()-t0
        prediction = dict(target_keys=list(features.targets), log_probabilities_base2=lp)
        if not np.isfinite(lp).all() or abs(np.exp2(lp).sum()-1) > 1e-5:
            raise ValueError('Invalid reader distribution')
        small = {k: v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                 for k, v in batch.items() if k not in IGNORED_BATCH_FIELDS}
        state = ReaderState(key, hash_json(history['graph']), small, **captured,
                            sequence_metadata=metadata, node_lookup=features.node_lookup,
                            clock_lookup=features.clock_lookup, targets=list(features.targets),
                            prior=list(features.prior), prediction=prediction,
                            timing=dict(preprocess_seconds=preprocess, model_seconds=elapsed),
                            provenance=list(provenance))
        state.edge_map()  # Fail before interpreting a nonunique edge alignment.
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                pickle.dump(state, handle, protocol=5)
            temporary.replace(path)
        return state

    @contextmanager
    def _pools(self, local, external):
        evidence = self.model.backbone.evidence
        native = evidence.forward
        local, external = local.to(self.device), external.to(self.device)
        evidence.forward = lambda batch: (local, external)
        try:
            yield
        finally:
            evidence.forward = native

    def predict_state(self, state, local=None, external=None, nodes=None, edges=None, detail='reader_A_hybrid'):
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in state.batch.items()}
        if nodes is not None:
            batch['nodes'] = nodes.to(self.device)
        if edges is not None:
            for key in EDGE_FIELDS:
                batch[key] = edges.batch[key].to(self.device)
        self.ledger.reserve('model_forward', detail=detail)
        with torch.inference_mode(), self._pools(state.local if local is None else local,
                                               state.external if external is None else external):
            lp = self.model(batch)[0, :len(state.targets)].cpu().tolist()
        return dict(target_keys=list(state.targets), log_probabilities_base2=lp)

    def audit(self, base, changed):
        if (base.graph_identity != changed.graph_identity or base.targets != changed.targets
                or base.prior != changed.prior or base.node_lookup != changed.node_lookup
                or base.clock_lookup != changed.clock_lookup):
            raise AssertionError('Graph/targets/legal prior/node mapping changed')
        all_columns = set(range(base.batch['nodes'].shape[1]))
        untouched = sorted(all_columns-set(COUNT_COLUMNS))
        if not torch.equal(base.batch['nodes'][:, untouched], changed.batch['nodes'][:, untouched]):
            raise AssertionError('Unassigned node fields changed')
        for key in set(base.batch)-{'nodes', *EDGE_FIELDS}:
            a, b = base.batch[key], changed.batch[key]
            same = torch.equal(a, b) if torch.is_tensor(a) else a == b
            if not same:
                raise AssertionError(f'Unassigned consumed tensor changed: {key}')
        a, b = base.edge_map(), changed.edge_map()
        fixed_a = {k: v for k, v in a.items() if k[-1] != 6}
        fixed_b = {k: v for k, v in b.items() if k[-1] != 6}
        if fixed_a.keys() != fixed_b.keys() or any(fixed_a[k][3] != fixed_b[k][3] for k in fixed_a):
            raise AssertionError('Physical topology or nominal edge metadata changed')
        count_diffs = []
        inverse = {i: n for n, i in {**base.node_lookup, **base.clock_lookup}.items()}
        for node in range(len(base.batch['nodes'])):
            for col in COUNT_COLUMNS:
                before, after = (float(s.batch['nodes'][node,col]) for s in (base,changed))
                if before != after:
                    count_diffs.append(dict(node=inverse[node],column=col,before=before,after=after))
        return dict(unassigned_tensors_identical=True, count_columns=list(COUNT_COLUMNS),
                    count_changes=count_diffs, P_local_changed=not torch.equal(base.local,changed.local),
                    count_fields_before={inverse[i]:base.batch['nodes'][i,list(COUNT_COLUMNS)].tolist() for i in range(len(inverse))},
                    count_fields_after={inverse[i]:changed.batch['nodes'][i,list(COUNT_COLUMNS)].tolist() for i in range(len(inverse))},
                    raw_sequence_metadata_before=base.sequence_metadata,
                    raw_sequence_metadata_after=changed.sequence_metadata,
                    P_external_changed=not torch.equal(base.external,changed.external),
                    edge_added=[list(k) for k in sorted(b.keys()-a.keys())],
                    edge_added_features=[dict(key=list(k),features=b[k]) for k in sorted(b.keys()-a.keys())],
                    edge_removed=[list(k) for k in sorted(a.keys()-b.keys())],
                    edge_removed_features=[dict(key=list(k),features=a[k]) for k in sorted(a.keys()-b.keys())],
                    edge_changed=[dict(key=list(k),before=a[k],after=b[k]) for k in sorted(a.keys() & b.keys()) if a[k]!=b[k]],
                    edge_identity='public endpoint kind/id + directed relation type; absence retained')

    def factorial(self, base, changed):
        audit = self.audit(base, changed)
        predictions = {}
        for p in (0, 1):
            for c in (0, 1):
                for e in (0, 1):
                    nodes = base.batch['nodes'].clone()
                    if c:
                        nodes[:, COUNT_COLUMNS] = changed.batch['nodes'][:, COUNT_COLUMNS]
                    ps = changed if p else base
                    predictions[f'{p}{c}{e}'] = self.predict_state(base, ps.local, ps.external,
                                                                 nodes, changed if e else base)
        audit['base_reconstruction'] = prediction_error(predictions['000'], base.prediction)
        audit['changed_reconstruction'] = prediction_error(predictions['111'], changed.prediction)
        return predictions, audit

    def predict_variant(self, history, variant='original', provenance=None):
        from .reader_candidates import predict_variant_state
        return predict_variant_state(self, self.prepare(history, provenance), variant)
