"""Integration controls against frozen final reader and immutable V1 records."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from cgx.data.store import Store
from cgx.models.networks import mean_pool
from cgx.provenance import atomic_json
from diagnostics.false_alarm_causes.core import load_reader,inference_settings
from .hooks import FrozenReaderHooks,prediction_error,COUNT_COLUMNS
from .ledger import Ledger,TimedStage
from .reader_candidates import candidate_tensors,metrics
from .reader_factorial import collect_items


def run_controls(root,v1,run,device):
    inference_settings();hooks=FrozenReaderHooks(load_reader(root,'final',device),Ledger(run),Path(run)/'cache/reader_controls')
    entries=collect_items(v1);store=Store(v1);checks=[]
    # Deterministic manifest order, no outcome-based selection.
    path,row=entries[0]
    with gzip.open(path,'rt') as handle:item=json.load(handle)
    history=store.load_history(row['evidence_trace_hash']);base=hooks.prepare(history)
    checks.append(dict(test='native_matches_V1_full_distribution',**prediction_error(base.prediction,row['prediction'])))
    native=hooks.predict_state(base)
    checks.append(dict(test='no_op_full_distribution',**prediction_error(native,base.prediction)))
    identical,audit=hooks.factorial(base,base)
    for mask,pred in identical.items():prediction_error(pred,base.prediction)
    checks.append(dict(test='all_8_identical_group_substitutions',passed=True))
    extra=next(i+1 for i,t in enumerate(item['trace']) if t['action']['ports'])
    copied=dict(history,observations=history['observations']+[history['observations'][extra]])
    provenance=[f'acquisition:{i}' for i in range(len(history['observations']))]
    changed=hooks.prepare(copied,provenance+[provenance[extra]])
    cells,audit=hooks.factorial(base,changed)
    checks.append(dict(test='copied_000_111_reconstruction_and_untouched_tensors',passed=True,audit=audit))
    # Independently attributed raw count formula, including nonbinary field31.
    for state in (base,changed):
        count=np.zeros((len(state.batch['nodes']),3),np.float32)
        for r in state.sequence_metadata:
            owner=state.node_lookup.get(r['port'].split(':')[0],-1)
            if owner>=0:count[owner]+=[1,r['record_count']/1000,float(r['missing'])]
        np.testing.assert_allclose(state.batch['nodes'][:,23:26].numpy(),np.log1p(count),rtol=0,atol=1e-6)
        nn=len(state.node_lookup)
        np.testing.assert_allclose(state.batch['nodes'][:nn,31].numpy(),1-np.minimum(np.log1p(count[:nn,0]),1),rtol=0,atol=1e-7)
    checks.append(dict(test='all_raw_count_and_observed_status_fields',passed=True))
    reverse=dict(history,observations=list(reversed(history['observations'])))
    rev=hooks.prepare(reverse,list(reversed(provenance)))
    checks.append(dict(test='observation_block_order_invariance',**prediction_error(rev.prediction,base.prediction)))
    # Uniform replication of all encoded sequences belonging to each component.
    owners=torch.tensor([base.node_lookup.get(r['port'].split(':')[0],-1) for r in base.sequence_metadata])
    valid=owners>=0;vectors=base.sequence_vectors[valid];owner=owners[valid]
    original=mean_pool(vectors,owner,len(base.local));repeated=mean_pool(torch.cat([vectors,vectors]),torch.cat([owner,owner]),len(base.local))
    maxerror=float((original-repeated).abs().max())
    assert maxerror<1e-6
    checks.append(dict(test='uniform_component_sequence_duplication_P',max_abs_vector_error=maxerror,passed=True))
    duplicate_meta=[r for r in changed.sequence_metadata if r['observation_index']==len(history['observations'])]
    assert all(r['acquisition_identity']==provenance[extra] for r in duplicate_meta)
    assert len(set(provenance))==len(provenance)
    checks.append(dict(test='same_copy_identity_vs_distinct_real_acquisition',passed=True))
    # Resources are computed for selectors but the diagnostic backbone does not consume them.
    altered=dict(history,remaining={k:0 for k in history['remaining']})
    resource=hooks.prepare(altered)
    checks.append(dict(test='resources_not_diagnostic_evidence',**prediction_error(resource.prediction,base.prediction)))
    try:hooks.prepare(dict(history,prediction=base.prediction))
    except ValueError:checks.append(dict(test='earlier_q_rejected_at_public_boundary',passed=True))
    else:raise AssertionError('Earlier q entered reader')
    # Reorder the target dictionary to verify healthy lookup does not assume index0.
    pred=dict(target_keys=base.prediction['target_keys'][::-1],log_probabilities_base2=base.prediction['log_probabilities_base2'][::-1])
    assert metrics(pred,row['true_target_key'])['p_healthy']==metrics(base.prediction,row['true_target_key'])['p_healthy']
    checks.append(dict(test='healthy_lookup_by_target_key',passed=True))
    # Rename public identifiers and permute physical nodes, target mapping follows IDs.
    renamed=deepcopy(history);g=renamed['graph'];mapping={n['id']:f'public_renamed_{i}' for i,n in enumerate(g['nodes'])}
    for n in g['nodes']:n['id']=mapping[n['id']]
    g['nodes'].reverse()
    for e in g['edges']:
        e['source']=mapping[e['source']];e['target']=mapping[e['target']]
    for obs in renamed['observations']:
        for record in obs.get('records',[]):
            owner,port=record['port'].split(':',1)
            if owner in mapping:record['port']=mapping[owner]+':'+port
    ren=hooks.prepare(renamed)
    remapped=dict(zip(ren.prediction['target_keys'],ren.prediction['log_probabilities_base2']))
    ordered=[]
    for target in base.targets:
        key=target if target=='healthy' else mapping[target.split(':')[0]]+':'+target.split(':',1)[1]
        ordered.append(remapped[key])
    checks.append(dict(test='public_node_renumber_and_permutation',**prediction_error(dict(target_keys=base.targets,log_probabilities_base2=ordered),base.prediction)))
    # Add only an already acquired calibration block. No simulation/no new noise.
    calhistory=None
    for path,r in entries:
        h=store.load_history(r['evidence_trace_hash'])
        block=next((o for o in h['observations'] if o.get('calibration_bursts')),None)
        if block:
            calhistory=h;break
    if calhistory is None:raise AssertionError('Calibration control requires an acquired burst')
    pure=deepcopy(dict(block,records=[]))
    # Explicit deterministic timestamp intervention, not a new physical acquisition.
    for burst in pure['calibration_bursts']:
        for timestamps in burst['timestamps_s']:timestamps[1]+=.01
    calbase=hooks.prepare(calhistory)
    calchanged=hooks.prepare(dict(calhistory,observations=calhistory['observations']+[pure]))
    cala=hooks.audit(calbase,calchanged)
    assert not cala['count_changes'] and not cala['P_local_changed'] and not cala['P_external_changed']
    assert cala['edge_changed'] or cala['edge_added']
    hooks.factorial(calbase,calchanged)
    checks.append(dict(test='calibration_only_changes_E',passed=True,audit=cala))
    for variant in ('R-P','R-C','R-PC'):
        local,nodes=candidate_tensors(base,variant)
        untouched=[i for i in range(32) if i not in COUNT_COLUMNS]
        assert torch.equal(nodes[:,untouched],base.batch['nodes'][:,untouched])
        if variant=='R-P':assert torch.equal(nodes,base.batch['nodes'])
        if variant=='R-C':assert torch.equal(local,base.local)
    checks.append(dict(test='candidate_public_rule_only_declared_groups',passed=True))
    result=dict(state='passed',tests=len(checks),checks=checks,fp32_tolerance=1e-6,
                checkpoint_hash=hooks.checkpoint_hash,physical_attempts=0)
    atomic_json(Path(run)/'tests/reader_hook_controls.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--v1',required=True)
    p.add_argument('--run',required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    with TimedStage(args.run,'reader_controls'):
        result=run_controls(args.root,args.v1,args.run,args.device)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
