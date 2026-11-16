from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
import pytest
from .events import prepare_history, annotate_history, canonical_evidence
from .likelihood import timing_normal_equations, integrate_gaussian, LIKELIHOOD_VERSION
from .reader import StructuredEvidenceDiagnoser
from cgx.provenance import hash_json


def parameters():
    mixture=dict(weights=[1.],means=[.05],variances=[.0001])
    return dict(version=LIKELIHOOD_VERSION,probability_floor=1e-8,
        timestamp_noise_variance=1e-6,calibration_noise_variance=1e-6,
        clock_prior_variance=1e-4,timing_process_variance={'cache_stale':2e-7,'queue_hold':3e-7},
        healthy_gain_mean=0.,gain_noise_variance=1e-10,
        gain_mixture=dict(weights=[1.],means=[.1],variances=[.001]),
        delay_mixtures={'cache_stale':mixture,'queue_hold':mixture})


def history():
    graph=json.loads((Path(__file__).parents[1]/'examples/public_graph.json').read_text())
    records=[]
    for node in graph['nodes']:
        if node['type']!='cache':continue
        for i,port in enumerate(node['ports']):
            records.append(dict(port=node['id']+':'+port['name'],local_time_s=1.+i*.02,
                message_token='abcdef',event_type='receive' if port['direction']=='in' else 'send',
                values=[.3]*port['width'],missing=False))
    return dict(graph=graph,observations=[dict(records=records,calibration_bursts=[],visible_status='ok')],
        action_keys=['ignored'],remaining={'executions':4},initial_budget={'executions':4})


def test_duplicate_raw_record_and_block_packaging():
    h=annotate_history(history(),{'observation_ids':['actual-acquisition-1']})
    r=StructuredEvidenceDiagnoser(parameters())
    expected=r.predict(h)['log_probabilities_base2']
    duplicate=deepcopy(h);duplicate['observations'][0]['records']*=4
    np.testing.assert_array_equal(r.predict(duplicate)['log_probabilities_base2'],expected)
    split=deepcopy(h);obs=split['observations'][0]
    split['observations']=[{**obs,'records':[record]} for record in reversed(obs['records'])]
    np.testing.assert_array_equal(r.predict(split)['log_probabilities_base2'],expected)


def test_numeric_equality_is_not_identity():
    h=history();h['observations']*=2
    evidence=prepare_history(h,{'observation_ids':['one','two']})
    assert len(evidence['events'])==2*len(h['observations'][0]['records'])
    repeated=prepare_history(h,{'observation_ids':['one','one']})
    assert len(repeated['events'])*2==len(evidence['events'])


def test_conflicting_identity_rejected():
    h=annotate_history(history(),{'observation_ids':['one']})
    record=deepcopy(h['observations'][0]['records'][0]);record['local_time_s']+=1
    h['observations'][0]['records'].append(record)
    with pytest.raises(ValueError,match='Conflicting'):
        prepare_history(h)


def test_calibration_numeric_equality_has_distinct_actual_ids():
    h=history();clocks=h['graph']['clock_domains']
    obs=dict(records=[],visible_status='ok',calibration_bursts=[dict(pair=clocks[:2],
        timestamps_s=[[1.,1.01,1.02,1.03]]*4)])
    h['observations']=[obs,deepcopy(obs)]
    assert len(prepare_history(h,{'observation_ids':['one','two']})['calibrations'])==8
    assert len(prepare_history(h,{'observation_ids':['one','one']})['calibrations'])==4
    with pytest.raises(ValueError,match='provenance'):
        prepare_history(h)


def test_action_and_resource_metadata_not_evidence():
    h=history();p={'observation_ids':['one']};r=StructuredEvidenceDiagnoser(parameters())
    expected=r.predict(h,p)['log_probabilities_base2']
    h['action_keys']=['arbitrary']*100;h['remaining']={'executions':0};h['prediction']={'healthy':1.}
    np.testing.assert_array_equal(r.predict(h,p)['log_probabilities_base2'],expected)


def test_exact_epoch_and_shared_clock_integral():
    p=dict(meta={'clocks':['a','b'],'delay_nodes':['cache']},
        times=np.array([1.,1.1,1.2,5.,5.13]),clocks=np.array([0,1,1,0,1]),
        groups=np.array([0,0,0,1,1]),group_count=2,
        delay_path=np.array([[0.],[0.],[1.],[0.],[1.]]),calibration=np.empty((0,3)))
    pars=parameters();pars['timestamp_noise_variance']=.04;pars['timing_process_variance']['cache_stale']=.01
    S,b,q,c=timing_normal_equations(p,pars,'cache','cache_stale')
    mu=np.array([0.,0.,.05]);prior=np.array([.2,.2,.03])
    actual,_,_=integrate_gaussian(S,b,q,c,mu,prior)
    contrast=np.array([[-1,1,0,0,0],[-1,0,1,0,0],[0,0,0,-1,1]])
    design=np.c_[np.eye(2)[p['clocks']],p['delay_path']]
    R=np.diag(.04+.01*p['delay_path'][:,0])
    covariance=contrast@(R+design@np.diag(prior)@design.T)@contrast.T
    expected=multivariate_normal.logpdf(contrast@p['times'],mean=contrast@design@mu,cov=covariance)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_shared_bias_does_not_vanish_with_repetition():
    noise=.01;shared=.04;n=1000
    assert noise/n+shared>.039
    assert (noise+shared)/n<.0001
    y=np.array([.2,.22,.19]);A=np.ones((3,1));S=A.T@A/noise;b=A.T@y/noise;q=y@y/noise
    c=-len(y)/2*np.log(2*np.pi*noise)
    actual,_,_=integrate_gaussian(S,b,q,c,[0.],[shared])
    expected=multivariate_normal.logpdf(y,mean=np.zeros(3),cov=noise*np.eye(3)+shared*np.ones((3,3)))
    np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_no_measurements_returns_legal_prior():
    h=history();h['observations']=[]
    r=StructuredEvidenceDiagnoser(parameters());prediction=r.predict(h)
    from cgx.models.features import target_space
    np.testing.assert_allclose(np.exp2(prediction['log_probabilities_base2']),target_space(h['graph'])[3],atol=1e-14)


def test_opaque_identity_renaming_is_not_a_feature():
    h=annotate_history(history(),{'observation_ids':['one']})
    e=prepare_history(h);r=StructuredEvidenceDiagnoser(parameters())
    expected=r.predict_evidence(e)['log_probabilities_base2']
    for i,event in enumerate(e['events']):
        event['measurement_event_id']=event['message_event_id']=f'renamed-{100-i}'
        event['execution_id']='a different opaque execution string'
        event['message_token']='a different opaque message string'
    np.testing.assert_allclose(r.predict_evidence(e)['log_probabilities_base2'],expected,atol=1e-10,rtol=0.)


def test_cached_acquisitions_preserve_shared_clock_joint_likelihood():
    h=history();second=deepcopy(h['observations'][0])
    for record in second['records']:record['local_time_s']+=.003
    h['observations'].append(second)
    clocks=h['graph']['clock_domains']
    h['observations'].append(dict(records=[],calibration_bursts=[dict(pair=clocks[:2],
        timestamps_s=[[1.,1.01,1.02,1.03],[3.,3.01,3.02,3.03]])],visible_status='ok'))
    p={'observation_ids':['one','two','calibration'],
       'observation_refs':[hash_json(o) for o in h['observations']]}
    reader=StructuredEvidenceDiagnoser(parameters())
    expected=reader.predict_evidence(prepare_history(h,p))
    for _ in range(2):
        actual=reader.predict(h,p)
        np.testing.assert_allclose(actual['log_probabilities_base2'],expected['log_probabilities_base2'],atol=1e-9,rtol=0.)
        np.testing.assert_allclose(actual['structured_diagnostics']['clock_covariance'],
            expected['structured_diagnostics']['clock_covariance'],atol=1e-12,rtol=0.)
    assert len(reader._acquisition_cache)==3
    assert actual['structured_diagnostics']['unique_calibration_events']==2


def test_cache_equal_values_new_identity_and_duplicate_storage():
    h=history();h['observations']*=2;reader=StructuredEvidenceDiagnoser(parameters())
    ref=hash_json(h['observations'][0])
    same=reader.predict(h,{'observation_ids':['one','one'],'observation_refs':[ref,ref]})
    fresh=reader.predict(h,{'observation_ids':['one','two'],'observation_refs':[ref,ref]})
    assert fresh['structured_diagnostics']['unique_measurement_events']==2*same['structured_diagnostics']['unique_measurement_events']
    np.testing.assert_allclose(fresh['log_probabilities_base2'],reader.predict_evidence(
        prepare_history(h,{'observation_ids':['one','two']}))['log_probabilities_base2'],atol=1e-9,rtol=0.)


def test_cache_annotated_split_packages_use_canonical_event_union():
    h=annotate_history(history(),{'observation_ids':['one']})
    reader=StructuredEvidenceDiagnoser(parameters());expected=reader.predict(h)
    obs=h['observations'][0]
    h['observations']=[{**obs,'records':[r]} for r in reversed(obs['records'])]
    actual=reader.predict(h,{'observation_ids':['one']*len(h['observations'])})
    np.testing.assert_array_equal(actual['log_probabilities_base2'],expected['log_probabilities_base2'])
    assert len(reader._acquisition_cache)==0
