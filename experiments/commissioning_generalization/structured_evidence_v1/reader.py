"""Frozen Structured Evidence Diagnoser; public observations enter one-way only."""
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import json
import numpy as np
from cgx.provenance import file_sha256, hash_json
from .events import prepare_history
from .likelihood import LIKELIHOOD_VERSION, prepare_evidence, combine_prepared_acquisitions, posterior


class StructuredEvidenceDiagnoser:
    def __init__(self,parameters,checkpoint_hash=None,training_configurations=()):
        if parameters['version']!=LIKELIHOOD_VERSION:raise ValueError('Unknown likelihood version')
        self.parameters_dict=deepcopy(parameters)
        self.checkpoint_hash=checkpoint_hash or hash_json(parameters)
        self.training_configurations=list(training_configurations)
        self.identity=dict(model='Structured Evidence Diagnoser',checkpoint_hash=self.checkpoint_hash,
            training_configurations=self.training_configurations,
            acquisition_counts_as_fault_features=False,shared_clock_offsets=True)
        self.floor=float(parameters.get('probability_floor',1e-8))
        self._acquisition_cache=OrderedDict()
        self._acquisition_cache_limit=64

    def prepare(self,history,provenance=None):
        provenance=provenance if provenance is not None else history.get('structured_provenance')
        observations=history.get('observations',[])
        if provenance is None or not observations:
            return prepare_evidence(prepare_history(history,provenance))
        ids=provenance['observation_ids']
        if len(ids)!=len(observations):raise ValueError('Observation provenance does not align with acquired blocks')
        refs=provenance.get('observation_refs')
        if refs is not None and len(refs)!=len(observations):raise ValueError('Observation references do not align')
        # Annotated event packages can overlap, split message groups or merge
        # acquisitions. Their explicit IDs always use the full canonical union.
        # Only complete legacy acquisition blocks enter the fast path.
        if any('execution_id' in o or
               any('measurement_event_id' in r or 'execution_id' in r for r in o.get('records',[])) or
               any('measurement_event_ids' in b or 'execution_id' in b for b in o.get('calibration_bursts',[]))
               for o in observations):
            return prepare_evidence(prepare_history(history,provenance))
        graph_hash=hash_json(history['graph']);seen={};blocks=[]
        for i,(acquisition,obs) in enumerate(zip(ids,observations)):
            acquisition=str(acquisition)
            # Optional references are immutable content-addressed Store refs,
            # supplied by the collector, not measurement identities. Equal
            # numerical contents with different actual IDs remain independent.
            content_ref=refs[i] if refs is not None else hash_json(obs)
            if acquisition in seen:
                if seen[acquisition]!=content_ref:
                    raise ValueError('Conflicting contents for the same acquisition identity')
                continue
            seen[acquisition]=content_ref
            key=(graph_hash,acquisition,content_ref)
            if key in self._acquisition_cache:
                prepared=self._acquisition_cache.pop(key)
            else:
                prepared=prepare_evidence(prepare_history(
                    dict(graph=history['graph'],observations=[obs]),{'observation_ids':[acquisition]}))
            self._acquisition_cache[key]=prepared
            if len(self._acquisition_cache)>self._acquisition_cache_limit:
                self._acquisition_cache.popitem(last=False)
            blocks.append(prepared)
        return combine_prepared_acquisitions(blocks)

    def predict_prepared(self,prepared,include_latents=True):
        return posterior(prepared,self.parameters_dict,include_latents)

    def predict_evidence(self,evidence,include_latents=True):
        return self.predict_prepared(prepare_evidence(evidence),include_latents)

    def predict(self,history,provenance=None,include_latents=True):
        return self.predict_prepared(self.prepare(history,provenance),include_latents)

    def predict_batch(self,histories,provenances=None,include_latents=False):
        if provenances is None:provenances=[None]*len(histories)
        return [self.predict(h,p,include_latents) for h,p in zip(histories,provenances)]

    def eval(self):return self
    def to(self,device):return self
    def parameter_fingerprint(self):return hash_json(self.parameters_dict)

    def shared_error_curve(self,evidence,repetitions=(1,2,4,8,16,32,64)):
        """Real acquired cross-clock contrasts, plus a labeled Gaussian debug curve.

        This is uncertainty in an uncalibrated average transit measurement,
        marginalizing the learned common clock prior. It is NOT a claim that
        identifiable posterior clock uncertainty can never shrink.
        """
        from collections import defaultdict
        from .events import canonical_evidence
        e=canonical_evidence(evidence);groups=defaultdict(lambda:defaultdict(list))
        for r in e['events']:
            if r['missing'] or r['port_id']=='external:out':continue
            groups[r['execution_id'],r['message_token']][r['port_id']].append(r)
        contrasts=defaultdict(list)
        for _,ports in sorted(groups.items()):
            keys=sorted(ports)
            for i,a in enumerate(keys):
                for z in keys[i+1:]:
                    ra=ports[a][0];rz=ports[z][0]
                    if ra['clock_group_id']==rz['clock_group_id']:continue
                    contrasts[a,z,ra['clock_group_id'],rz['clock_group_id']].append(rz['timestamp']-ra['timestamp'])
        if not contrasts:return dict(state='no_acquired_cross_clock_pair',curves=[])
        pair=max(sorted(contrasts),key=lambda p:len(contrasts[p]));values=np.asarray(contrasts[pair])
        independent=2*float(self.parameters_dict['timestamp_noise_variance'])
        shared=2*float(self.parameters_dict['clock_prior_variance'])
        ns=sorted(set(min(int(n),len(values)) for n in repetitions))
        rows=[dict(n=n,actual_mean_difference_seconds=float(values[:n].mean()),
            independent_variance=independent/n,shared_clock_variance=shared,
            correct_marginal_average_variance=independent/n+shared,
            incorrect_independent_clock_average_variance=(independent+shared)/n)
            for n in ns]
        return dict(state='passed',ports=list(pair[:2]),clock_groups=list(pair[2:]),
            available_unique_message_contrasts=len(values),curves=rows,
            hypothesis='learned healthy timing noise, uncalibrated shared-clock prior',
            no_new_physical_measurements_generated=True,
            conditional_clock_posterior_may_shrink_when_identifiable=True)


def load_reader(run_dir,name='final',device='cpu'):
    directory=Path(run_dir)/'models/readers'/name
    marker=json.loads((directory/'complete.json').read_text())
    if marker['state']!='passed':raise RuntimeError('Structured reader fit is not complete')
    path=directory/marker.get('selected_checkpoint','model.json')
    digest=file_sha256(path)
    if digest!=marker['selected_checkpoint_hash']:raise RuntimeError('Structured reader checkpoint changed')
    payload=json.loads(path.read_text())
    return StructuredEvidenceDiagnoser(payload['parameters'],digest,payload['training_configurations'])
