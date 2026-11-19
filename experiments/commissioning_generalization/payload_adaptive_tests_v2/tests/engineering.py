"""Executable scientific boundary checks. Run on a source object only."""
import copy
import inspect
import json
from pathlib import Path
import time
import numpy as np
from payload_change_v1.estimator import PayloadEvidenceEstimator,payload_to_pi
from payload_adaptive_tests_v2.protocol import Ledger,write_json,digest,verify_protected
from payload_adaptive_tests_v2.worker import new_plant,context,save_block
from payload_adaptive_tests_v2.cases import policy_view
from payload_adaptive_tests_v2.trajectories import b0,library,scaled,reference
from payload_adaptive_tests_v2.evaluate import setup
from payload_adaptive_tests_v2.prediction import Clock
from payload_adaptive_tests_v2.policies import Selector
from payload_adaptive_tests_v2.fixed_plans import diverse_order
from payload_adaptive_tests_v2.physics import _metric

def run(root):
    root=Path(root); case=json.loads((root/'private/source.json').read_text())[0]
    results=[]; cpu=Ledger(root,'cpu'); token=cpu.reserve(600,unit='engineering_suite',kind='engineering_tests'); begin=time.process_time()
    def check(name,condition,details=None):
        results.append(dict(name=name,passed=bool(condition),details=details))
        write_json(root/'checks/engineering_tests.json',dict(passed=all(x['passed'] for x in results),checks=results,complete=False))
        if not condition: raise AssertionError(name)
    check('parent_preservation',verify_protected(root)['passed'])
    pub=policy_view(case); swapped=copy.deepcopy(case); swapped.update(group='permuted',true_payload={'forbidden':'value'},family_id='other')
    check('no_private_leakage_public_view',policy_view(swapped)==pub)
    ctx,est,pred,bank,_=setup(root,pub,'adaptive_work_decision')
    check('selection_API_no_plant', 'plant' not in inspect.signature(Selector.choose).parameters and not any('plant' in k for k in pred.__dict__))
    plant=new_plant(root,case,5,tag='engineering'); bias=plant._bias.copy(); ar=plant._ar.copy()
    b=save_block(root,plant,b0(),Path('raw/engineering/B0'),'engineering')
    check('persistent_bias',np.array_equal(bias,plant._bias) and not np.array_equal(ar,plant._ar))
    before=plant.cache_identity(); clone=plant.clone()
    check('complete_clone_identity',before==clone.cache_identity())
    x=plant.execute(library()[0],'engineering_shared_branch',kind='engineering_clone')
    y=clone.execute(library()[0],'engineering_shared_branch',kind='engineering_clone')
    x.save(root/'raw/engineering/clone_a'); y.save(root/'raw/engineering/clone_b')
    check('clone_bitwise_prefix_and_continuation',all(np.array_equal(x.public[k],y.public[k]) for k in x.public))
    check('no_reset_state_continuity',b.metadata['end_state']==x.metadata['start_state'])
    true_hash=plant._plant_spec_hash; masses=plant._model.body_mass.copy()
    newcfg=copy.deepcopy(case['configured_payload']); newcfg['mass']*=1.05
    plant.configure(newcfg)
    check('configuration_only_effect',true_hash==plant._plant_spec_hash and np.array_equal(masses,plant._model.body_mass))
    modified=copy.deepcopy(case); modified['true_payload']['mass']*=1.10
    modified['true_payload']['inertia_com']=(np.asarray(modified['true_payload']['inertia_com'])*1.10).tolist()
    other=new_plant(root,modified,5,tag='engineering')
    mb=save_block(root,other,b0(),Path('raw/engineering/changed_payload'),'engineering_payload_effect')
    check('actual_payload_effect',np.max(abs(mb.public['q']-b.public['q']))>1e-5 and np.max(abs(mb.public['torque']-b.public['torque']))>1e-3)
    est.ingest(b.public|dict(execution_id='first'))
    m,c,_=est.posterior(); n=len(est._records)
    est.ingest(b.public|dict(execution_id='duplicate_wrapper'))
    m2,c2,_=est.posterior()
    check('duplicate_event_invariance',np.array_equal(m,m2) and np.array_equal(c,c2) and len(est._records)==n)
    est.ingest(x.public|dict(execution_id='second'))
    check('independent_event_identity',len(est._records)==n+len(x.public['q']) and not np.array_equal(b.public['event_id'][:10],x.public['event_id'][:10]))
    for a in library():
        for s in (.75,.5):
            for t in (.2,2.3,5.7,7.9):
                q,v,ac=reference(a,t); qs,vs,acs=reference(scaled(a,s),t/s)
                check('time_scaling_'+a['id']+'_'+str(s)+'_'+str(t),np.allclose(q,qs) and np.allclose(v*s,vs) and np.allclose(ac*s*s,acs))
    order=diverse_order(bank,m,c,est.parameters['posterior_covariance_inflation'])
    cand=[dict(base_id=order[0]),dict(base_id=order[1])]
    for method in ('fixed_prior_plan','plan_after_b0'):
        select=Selector(method,bank,pub['configured_payload'],1.); select.order=order.copy(); select.fallback_order=order.copy()
        a,_,_=select.choose(m,c,cand,pred,[],Clock(10)); alt=m.copy(); alt[0]+=.4
        bsel,_,_=select.choose(alt,c*2,cand,pred,[],Clock(10)); blocked,_,_=select.choose(alt,c*2,cand[1:],pred,[],Clock(10))
        check(method+'_fixed_ranking_live_gate',a==bsel and blocked==cand[1] and select.order==order)
    k=pred.kernel(case['configured_payload'],b0(),clock=Clock(10))
    hypothesis=copy.deepcopy(case['configured_payload']); hypothesis['mass']*=1.01
    hypothesis['inertia_com']=(np.asarray(hypothesis['inertia_com'])*1.01).tolist()
    real=pred._integrate(hypothesis,case['configured_payload'],b0(),np.asarray(b.metadata['start_state']['q'])*0+__import__('cgx.env.system',fromlist=['HOME']).HOME,False,Clock(10))
    linear=k['ee']+np.einsum('tij,j->ti',k['jee'],payload_to_pi(hypothesis)-k['configured_pi'])
    error=float(np.max(np.linalg.norm(real['ee']-linear,axis=1)))
    check('public_linearized_tracking_against_independent_model',error<.0002,dict(max_error_m=error))
    check('prediction_not_acquired_evidence',len(est._records)==n+len(x.public['q']))
    q=np.zeros((500,7)); e=np.zeros(500); e[-100]=.003
    metric=_metric(q,q,e,np.zeros((500,7),bool),np.zeros(500),np.ones(7)*-1,np.ones(7),1.,False)
    check('full_final_hold_not_endpoint',not metric['final_hold_pass'] and metric['final_error_m']==0.)
    import payload_adaptive_tests_v2.evaluate as evaluate
    import payload_adaptive_tests_v2.shared_decision as shared
    import payload_adaptive_tests_v2.motion_screen as screen
    check('equal_wrapper_callables',evaluate.deployment_choices is shared.deployment_choices and evaluate.candidates is screen.candidates)
    check('budget_charges_complete_source_blocks',json.loads((root/'resources/totals.json').read_text())['plant']['completed']>=32)
    check('normal_half_separate',evaluate.summarize.__code__.co_consts.count('normal_correct')>0 or 'normal_correct' in inspect.getsource(evaluate.summarize))
    cpu.complete(token,time.process_time()-begin)
    result=dict(passed=all(x['passed'] for x in results),complete=True,checks=results)
    write_json(root/'checks/engineering_tests.json',result); return result

if __name__=='__main__':
    import sys
    print(json.dumps(run(sys.argv[1]),indent=2))
