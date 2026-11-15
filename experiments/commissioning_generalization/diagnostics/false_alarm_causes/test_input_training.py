"""Reproduce deterministic feature and reward-audit invariants without new physics."""
from copy import deepcopy
import os

from .input_checks import engineering_checks, history_metrics, _fixture
from .ledger import Ledger
from .training_audit import audit_rows
from cgx.training.engine import transition_key


def test_actual_feature_implementation_matches_analytic_records(tmp_path):
    ledger=Ledger(os.environ.get('CGX_FA_RUN',tmp_path))
    results=engineering_checks(ledger)
    assert results and all(r['passed'] for r in results), results


def fixture_rows():
    budget=dict(executions=2,calibrations=2,bytes=1000,decisions=4)
    current=budget.copy();rows=[];scores={}
    lps=[-3.,-4.,-1.,-1.]
    for i,kind in enumerate(['single_port','calibration_only','stop']):
        cost=dict(executions=int(i==0),calibrations=int(i==1),bytes=[256,160,0][i],decisions=int(i<2))
        nxt={k:v-cost[k] for k,v in current.items()}
        row=dict(trajectory_id='trajectory',step=i,split='train',configuration_id='source',parent_case_id='case',
                 source_fold=0,true_target_key='healthy',execution_cap=2,byte_fraction=1.,behavior='random_legal',
                 action_key=str(i),history_hash=str(i),next_history_hash=str(i+1 if i<2 else i),
                 action=dict(action_key=str(i),kind=kind,ports=['n1:out'] if i==0 else [],cost_bound=cost),
                 charged_cost=cost,resources_before=current,resources_after=nxt,terminal=i==2)
        rows.append(row);current=nxt
        scores[transition_key(row)]=dict(reward=lps[i+1]-lps[i],next_nll=-lps[i+1])
    return rows,scores


def test_source_audit_preserves_negative_reward_and_telescopes():
    rows,scores=fixture_rows()
    prefix,traj,_,counts,failures,_=audit_rows(rows,scores)
    assert not failures
    assert counts['telescoping_trajectories']==1
    assert traj[0]['total_reward_bits']==2.
    assert traj[0]['raw_records']==2
    assert prefix[-1]['calibrations']==1


def test_source_audit_detects_broken_masks_resources_and_rewards():
    rows,scores=fixture_rows();broken=deepcopy(rows)
    broken[0]['resources_after']['bytes']+=1
    broken[1]['terminal']=True
    scores[transition_key(broken[2])]['reward']=.2
    *_,counts,failures,groups=audit_rows(broken,scores)
    assert counts['resource_subtraction_failed']>=1
    assert counts['terminal_only_at_end_failed']==1
    assert counts['stop_zero_reward_failed']==1
    assert counts['telescoping_failed']==1


def test_history_metrics_excludes_initial_and_counts_duplicate_reading():
    history=_fixture()
    history['observations']*=3
    metrics=history_metrics(history)
    assert metrics['executions']==2
    assert metrics['bytes']==512
    assert metrics['raw_records']==4
    assert metrics['repeated_actions']==1
