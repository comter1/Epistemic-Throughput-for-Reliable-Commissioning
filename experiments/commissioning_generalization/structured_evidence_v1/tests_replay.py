"""Unit controls for conditional metrics, paired sampling, and raw packaging.

Synthetic values here test statistics and serialization mechanics only. Research
outcomes come exclusively from the existing physically acquired Panda histories.
"""
from copy import deepcopy
from types import SimpleNamespace
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from .events import annotate_history,prepare_history,evidence_identity
from .invariance import split_raw_blocks,merge_raw_blocks
from .replay import prediction_metrics,provenance_for_history
from .reporting import aggregate,paired_bootstrap,ece,main_cell_keys,timeout_annotations,timeout_stage_table
from .test_reader import history


def test_healthy_argmax_is_not_half_probability_threshold():
    p=dict(target_keys=['healthy','n0:cache_stale','n1:queue_hold','n2:adapter_scale'],
        log_probabilities_base2=np.log2([.4,.2,.2,.2]).tolist())
    row=prediction_metrics(p,'healthy')
    assert row['correct']==1 and row['healthy_false_alarm']==0 and row['p_healthy']==.4
    fault=prediction_metrics(p,'n0:cache_stale')
    assert fault['fault_miss']==1 and fault['joint_fault_correct']==0


def test_conditional_denominators_and_brier():
    keys=['healthy','n0:cache_stale'];rows=[]
    for i,(truth,prob) in enumerate([('healthy',[.8,.2]),('healthy',[.4,.6]),('n0:cache_stale',[.7,.3]),('n0:cache_stale',[.1,.9])]):
        rows.append(dict(case_id=str(i),configuration_id='a',policy_seed=11,
            **prediction_metrics(dict(target_keys=keys,log_probabilities_base2=np.log2(prob)),truth)))
    result=aggregate(pd.DataFrame(rows))
    assert result['rows']==4 and result['healthy_rows']==2 and result['fault_rows']==2
    assert result['false_alarm_count']==1 and result['healthy_false_alarm_rate']==.5
    assert result['miss_count']==1 and result['fault_miss_rate']==.5
    assert result['fault_joint_correct_count']==1 and result['fault_joint_accuracy']==.5
    assert result['accuracy']==.5
    assert np.isclose(rows[0]['brier'],.08)


def test_ece_fixed_bins_include_probability_one():
    assert ece([1.,.5],[1,0])==.25
    assert ece([.75,.75],[1,0])==.25


def test_unknown_failed_resources_keep_classification_denominator():
    p=dict(target_keys=['healthy','n0:cache_stale'],log_probabilities_base2=np.log2([.7,.3]))
    metric=prediction_metrics(p,'healthy')
    frame=pd.DataFrame([dict(metric,case_id='a',configuration_id='c',policy_seed=11,
        executions_used=4,resource_accounting_unavailable=False),
        dict(metric,case_id='b',configuration_id='c',policy_seed=11,
        executions_used=None,resource_accounting_unavailable=True)])
    result=aggregate(frame)
    assert result['rows']==2 and result['correct_count']==2
    assert result['mean_executions_used']==4
    assert result['executions_used_known_rows']==1 and result['executions_used_unknown_rows']==1
    assert result['resource_accounting_unavailable_count']==1


def test_full_recording_nullable_seed_has_stable_cell_identity():
    row=dict(suite='test_connections',method='full_recording_feasible',policy_seed=None,execution_cap=4,case_id='a')
    a=pd.DataFrame([row]);b=pd.DataFrame([{**row,'policy_seed':float('nan')}])
    assert main_cell_keys(a)==main_cell_keys(b)


def test_timeout_stage_provenance_distinguishes_selector_and_reader():
    selection=timeout_annotations(['selection_error:CallDeadlineExceeded:online limit'],True)
    assert selection['timeout_selection']==1 and selection['timeout_diagnostic']==0
    assert selection['timeout_any_from_errors']==1 and selection['timeout_unclassified']==0
    mixed=timeout_annotations(['fixed_plan_timeout','selection_warmup_error:TimeoutError:limit',
        'diagnostic_error:CallDeadlineExceeded:limit'],True)
    assert mixed['timeout_selection']==1 and mixed['timeout_diagnostic']==1
    assert mixed['timeout_any_from_errors']==1
    ordinary=timeout_annotations(['selection_error:ValueError:mentions TimeoutError in message'],False)
    assert ordinary['timeout_selection']==0 and ordinary['timeout_any_from_errors']==0


def test_initial_and_infrastructure_timeout_flag_disagreements_remain_visible():
    initial=timeout_annotations(['initial_execution_error:TimeoutError:failed'],False)
    assert initial['timeout_acquisition']==1 and initial['timeout_error_without_flag']==1
    worker=timeout_annotations(['worker_error:CallDeadlineExceeded:failed'],True)
    assert worker['timeout_infrastructure']==1
    absent=timeout_annotations([],True)
    assert absent['timeout_unclassified']==1 and absent['timeout_flag_without_error_stage']==1
    broken=timeout_annotations(['worker_error:ValueError:construction failed'],False)
    assert broken['timeout_infrastructure']==0


def test_timeout_stage_table_retains_denominator_and_correct_timeout_cases():
    rows=[]
    for case,errors,correct in [('a',['selection_timeout'],1),('b',[],0)]:
        rows.append(dict(version='structured',suite='test_connections',method='gnn_depth2',execution_cap=4,
            case_id=case,correct=correct,nll_bits=1.,**timeout_annotations(errors,bool(errors))))
    table=timeout_stage_table(pd.DataFrame(rows),['version','suite','method','execution_cap'])
    selected=table[table.timeout_stage=='selection'].iloc[0]
    assert selected.planned_rows==2 and selected.timeout_stage_rows==1
    assert selected.correct_among_stage==1 and selected.incorrect_among_stage==0
    assert selected.classification_denominator_retained


def _paired_frame():
    rows=[]
    for config in ('a','b','c'):
        for i,family in enumerate(('healthy','cache_stale','queue_hold','adapter_scale')):
            for seed in (11,23,37):
                rows.append(dict(configuration_id=config,case_id=config+str(i),fault_family=family,
                    policy_seed=seed,correct=int(i%2==0),is_healthy=i==0,
                    healthy_false_alarm=0,fault_miss=int(i==1),joint_fault_correct=int(i==2),
                    nll_bits=1.+i+seed*.01,brier=.3+i*.1))
    return pd.DataFrame(rows)


def test_paired_bootstrap_preserves_constant_delta_and_seed_unit():
    old=_paired_frame();new=old.copy();new['nll_bits']+=.25
    rows=paired_bootstrap(old,new,repetitions=100)
    nll=next(r for r in rows if r['metric']=='nll_bits')
    assert nll['physical_cases']==12
    assert np.isclose(nll['difference'],.25) and np.isclose(nll['ci_low'],.25) and np.isclose(nll['ci_high'],.25)
    repeated=paired_bootstrap(pd.concat([old,old]),pd.concat([new,new]),repetitions=100)
    assert repeated==rows


def test_paired_bootstrap_rejects_changed_physical_support():
    a=_paired_frame()
    with pytest.raises(ValueError,match='support'):
        paired_bootstrap(a,a[a.case_id!='a0'],repetitions=10)


def test_raw_file_packaging_preserves_identities_and_status():
    original=history();original['observations'][0]['visible_status']='timeout'
    annotated=annotate_history(original,{'observation_ids':['a']})
    baseline=evidence_identity(prepare_history(annotated))
    split=split_raw_blocks(annotated,pieces=7)
    assert len(split['observations'])==7
    assert evidence_identity(prepare_history(split))==baseline
    assert evidence_identity(prepare_history(merge_raw_blocks(split)))==baseline
    with pytest.raises(ValueError,match='stable ingress'):
        split_raw_blocks(original)


def test_acquisition_provenance_not_numeric_equality():
    h=history();obs=deepcopy(h['observations'][0]);obs.update(actual_cost={'executions':1,'calibrations':0},action_key='same-action')
    h['observations']=[deepcopy(obs),deepcopy(obs),deepcopy(obs)]
    ids=provenance_for_history(h,'case')['observation_ids']
    assert len(set(ids))==3
    h['observations'][1]['records'][0]['local_time_s']+=100
    assert provenance_for_history(h,'case')['observation_ids']==ids


@pytest.mark.skipif('CGX_REPLAY_RUN' not in os.environ,reason='Real saved-history inventory is an explicit integration-test input')
def test_rc_count_substitution_equals_preserved_v2_implementation():
    from . import replay
    from diagnostics.false_alarm_causes_v2.hooks import sequence_metadata
    from diagnostics.false_alarm_causes_v2.reader_candidates import candidate_tensors
    from cgx.data.store import Store
    path=Path(os.environ['CGX_REPLAY_RUN'])/'replay/jobs/00000.json.gz'
    job=replay.read_gzip(path)[0];replay._worker_init(os.environ['CGX_REPLAY_RUN'],'baselines')
    prepared=replay._prepare(job);features=prepared['features'];store=Store(job['store_root'])
    packed=job['packed'] or store.history_references(job['history_hash'])
    acquired=replay.materialize_packed(store,packed)
    state=SimpleNamespace(local=torch.zeros((len(features.nodes),1)),
        batch={'nodes':torch.from_numpy(features.nodes)},node_lookup=features.node_lookup,
        sequence_metadata=sequence_metadata(acquired))
    _,expected=candidate_tensors(state,'R-C')
    np.testing.assert_array_equal(prepared['corrected_nodes'],expected.numpy())
