"""V2 execution-axis and lossless performance regressions; artificial fixtures only."""
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Frozen scalar reference from the original sealed implementation.
def reference_bin_sequence(records, matches, direction, cap=256):
    """No record ordinal, ID value or common-time marker is encoded."""
    rows=[]; previous=None
    start=float(records[0]['local_time_s'])
    for r in records:
        values=np.zeros(14,dtype=np.float32)
        values[:len(r['values'])]=r['values']
        token=r['message_token']; time=float(r['local_time_s'])
        peers=matches.get(token,())
        other=[p for p in peers if p['port']!=r['port']]
        delta=np.asarray([p['local_time_s']-time for p in other],dtype=np.float32)
        rows.append((values,time-start,0.0 if previous is None else time-previous['local_time_s'],
                     float(previous is not None and previous['message_token']==token),
                     float(r['missing']),delta,float(bool(other)),
                     ('receive','send','sample').index(r['event_type'])))
        previous=r
    bins=[]
    for ids in np.array_split(np.arange(len(rows)),min(cap,len(rows))):
        vs=np.stack([rows[i][0] for i in ids]); v=np.zeros(64,dtype=np.float32)
        v[:14]=vs.mean(0); v[14:28]=vs.min(0); v[28:42]=vs.max(0)
        v[42:49]=vs[0,:7]; v[49:53]=vs[-1,:4]
        v[53]=direction
        v[54]=np.mean([rows[i][7]==0 for i in ids]);v[55]=np.mean([rows[i][7]==1 for i in ids])
        v[56]=rows[ids[0]][1]; v[57]=rows[ids[-1]][1]
        v[58]=np.mean([rows[i][2] for i in ids])
        v[59]=np.mean([rows[i][3] for i in ids])
        v[60]=np.mean([rows[i][4] for i in ids])
        deltas=[d for i in ids for d in rows[i][5]]
        v[61]=float(np.mean(deltas)) if deltas else 0.0
        v[62]=np.mean([rows[i][6] for i in ids])
        v[63]=math.log1p(len(ids))
        bins.append(v)
    return np.stack(bins)


def test_vectorized_encoder_preserves_frozen_scalar_output_bitwise():
    from cgx.models.features import _bin_sequence
    rng=np.random.default_rng(914)
    for n in (1,10,99,256,257,999,1000):
        records=[dict(port='n0:out',local_time_s=float(rng.normal()+i/100),values=rng.normal(size=14).tolist(),
            message_token='token'+str(i//3),missing=bool(i%13==0),event_type=('send','receive','sample')[i%3]) for i in range(n)]
        matches=defaultdict(list)
        for r in records:
            matches[r['message_token']].append(r)
            matches[r['message_token']].append({**r,'port':'n1:in','local_time_s':r['local_time_s']+.003})
        np.testing.assert_array_equal(reference_bin_sequence(records,matches,1),_bin_sequence(records,matches,1))


def test_v2_quality_crossing_uses_execution_cap_and_keeps_unreached():
    from cgx.reporting.diagnostics import cost_to_quality
    rows=[dict(suite='test_connections',method=m,policy_seed=11,byte_fraction=1.,execution_cap=cap,
        final_nll_bits=loss,bytes_used=cap*100,executions_used=cap,calibrations_used=0,all_cases_feasible=True)
        for m in ('a','b') for cap,loss in ((1,3.),(2,1. if m=='a' else 3.),(4,.5 if m=='a' else 3.))]
    result=cost_to_quality(pd.DataFrame(rows),dict(source='validation',rule='selected_final_reader_mean_validation_nll',budget_axis='execution_cap',target_nll_bits=2.))
    assert len(result)==2
    a=result[result.method=='a'].iloc[0];b=result[result.method=='b'].iloc[0]
    assert a.status=='reached' and a.execution_cap==2 and a.actual_executions==2
    assert b.status=='not_reached' and pd.isna(b.actual_executions)


def test_v2_four_main_figures_include_full_recording_and_three_seeds(tmp_path):
    from cgx.config import load_config
    from cgx.evaluation.design import matrix_cells
    from cgx.reporting.v2 import figures_v2
    from cgx.provenance import atomic_json
    config=load_config();out=tmp_path/'results';out.mkdir()
    rows=[];repairs=[]
    for suite in ('test_seen','test_connections','test_size'):
        for method,seed,cap,fraction in matrix_cells(config):
            rows.append(dict(suite=suite,method=method,policy_seed=seed,byte_fraction=fraction,execution_cap=cap,
                final_nll_bits=4.-cap*.2+(suite!='test_seen')*.1,correct_joint=.5,bytes_used=cap*100.,executions_used=cap,abstained=0.))
            if cap==4:repairs.append(dict(suite=suite,method=method,policy_seed=seed,mean_total_deployed_executions=7.,mean_total_deployed_bytes=500.,yield_fraction=.4))
    pd.DataFrame(rows).to_csv(out/'per_seed.csv',index=False);pd.DataFrame(repairs).to_csv(out/'repair_summary.csv',index=False)
    atomic_json(tmp_path/'models/thresholds.json',{'cost_to_quality':{'target_nll_bits':3.5}})
    atomic_json(out/'paired_statistics.json',[dict(primary=True,difference_nll_bits=.01,ci_low=-.1,ci_high=.2)])
    result=figures_v2(tmp_path,config)
    assert len(result['figures'])==8 and all(Path(p).stat().st_size>100 for p in result['figures'])
    gap=pd.read_csv(out/'V2_seen_unseen_gap.csv')
    assert len(gap)==16 and len(gap[gap.method=='full_recording_feasible'])==1
    assert '0.0100 bits' in Path(result['main_report']).read_text()


def test_builtin_reader_keeps_public_input_unchanged_and_custom_reader_is_copied():
    import torch
    from test_learning import fixture
    from cgx.models import DiagnosticReader
    from cgx.evaluation.rollout import reader_history
    from cgx.provenance import hash_json
    torch.set_num_threads(1);history,_=fixture();model=DiagnosticReader().eval();before=hash_json(history)
    model.predict(reader_history(model,history))
    assert hash_json(history)==before
    class ArbitraryReader:pass
    copied=reader_history(ArbitraryReader(),history)
    copied['observations'].clear()
    assert hash_json(history)==before and copied!=history


def test_replay_index_is_immutable_and_training_budgets_are_independent(tmp_path):
    import sqlite3,pytest
    from cgx.training.replay import Replay
    from cgx.data.store import Store
    from cgx.provenance import atomic_json
    from test_data import history
    store=Store(tmp_path);h=history();ref=store.put_history(h)
    # Explicit integrity fixture, not collected physical evidence.
    row=dict(parent_case_id='fixture',configuration_id='fixture-graph',source_fold=0,trajectory_id='fixture-trajectory',
        behavior='structure_rule',history_hash=ref,next_history_hash=ref,action={'kind':'stop'})
    digest=store.write_transitions('train','fixture',[row])
    atomic_json(store.completion_path('train','fixture'),dict(state='passed',technical_smoke=True,transition_file_hash=digest))
    replay=Replay(tmp_path)
    with pytest.raises(sqlite3.OperationalError):replay.db.execute("DELETE FROM transitions")
    a=replay.training_state(ref);a['remaining']['executions']=0;a['action_keys'].append('mutated')
    b=replay.training_state(ref)
    assert b['remaining']==h['remaining'] and b['action_keys']==[]
    ordinary=replay.state(ref);ordinary['graph']['nodes'].clear()
    assert replay.training_state(ref)['graph']['nodes']
