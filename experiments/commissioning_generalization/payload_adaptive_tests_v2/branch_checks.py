"""Source-only matched counterfactual continuations; never deployable resets."""
import copy
import json
from pathlib import Path
import time
import numpy as np
from .protocol import digest,write_json
from .cases import policy_view
from .worker import new_plant,save_block,stream
from .evaluate import setup,initial_motion,offline_candidates,run_policy
from .prediction import Clock,DecisionTimeout
from .motion_screen import candidates,class_alarms
from .shared_decision import deployment_choices,serialize_choice
from .policies import Selector
from .fixed_plans import diverse_order,post_cov

def selected_cases(root):
    all_cases=json.loads((Path(root)/'private/source.json').read_text()); selected=[]
    for ctx in ('W0','W1','W2'):
        for group in ('G0','G1','G2','G3'):
            selected+=sorted([c for c in all_cases if c['context']==ctx and c['group']==group],
                             key=lambda c:digest(['source_map_v2',c['case_id']]))[:2]
    return selected

def execute(root,case):
    root=Path(root); pub=policy_view(case); ctx,est,pred,bank,_=setup(root,pub,'adaptive_work_decision')
    prior_mean,prior_cov,_=est.posterior(); prior_est=est.estimate()['payload_estimate']
    fixed=Selector('fixed_prior_plan',bank,pub['configured_payload'],est.parameters['posterior_covariance_inflation'],2)
    clock=Clock(10)
    try:
        before,_,_=deployment_choices(pred,pub['configured_payload'],prior_est,np.asarray(ctx.get('home',__import__('cgx.env.system',fromlist=['HOME']).HOME)),prior_mean,prior_cov,clock,False,True)
        fixed.freeze(prior_mean,prior_cov,offline_candidates(bank),pred,before,clock)
    except DecisionTimeout: fixed.order=diverse_order(bank,prior_mean,prior_cov,fixed.inflation)
    plant=new_plant(root,case,5); action,initial_checks=initial_motion(pred,pub,Clock(10))
    b=save_block(root,plant,action,Path('raw/branches')/case['case_id']/'prefix','branch_common_B0')
    est.ingest(b.public|{'execution_id':b.metadata['execution_id']}); mean,cov,_=est.posterior()
    kinds={b.metadata['execution_id']:action['kind']}; start=b.public['q'][-1]
    choices,release,_=deployment_choices(pred,pub['configured_payload'],est.estimate()['payload_estimate'],start,mean,cov,Clock(10),est.estimate()['residual_consistent'],True)
    admitted,screens,timed_out=candidates(pred,bank,est.estimate()['payload_estimate'],start,mean,cov,60-b.physical_seconds,class_alarms(est,kinds),Clock(10))
    selectors={}; rankings=[]
    for method in ('fixed_prior_plan','plan_after_b0','simple_parameter_rule','adaptive_variance','adaptive_work_decision'):
        selector=fixed if method=='fixed_prior_plan' else Selector(method,bank,pub['configured_payload'],fixed.inflation)
        clock=Clock(10)
        try:
            chosen,scores,fallback=selector.choose(mean,cov,admitted,pred,choices,clock)
            selectors[method]=dict(chosen=chosen['base_id'] if chosen else None,order=selector.order,
                fallback_reason=fallback,decision_cpu_seconds=clock.used)
            rankings.extend(dict(case_id=case['case_id'],context=case['context'],method=method,**s) for s in scores)
        except DecisionTimeout:
            selectors[method]=dict(chosen=None,fallback_reason='source_map_timeout',decision_cpu_seconds=clock.used)
    decision=Selector('adaptive_work_decision',bank,pub['configured_payload'],fixed.inflation)
    rank_check=[]
    try:
        rank_check=decision.score(mean,cov,admitted,pred,choices,Clock(10),kind='decision',samples=256,outcomes=32)
    except DecisionTimeout: rank_check=[dict(reason='256_32_rank_check_timeout')]
    # Gaussian-value dependence unit: identical precision, different numerical outcomes.
    info=np.asarray(next(iter(bank['actions'].values()))['information17'])
    cp=post_cov(cov,info,fixed.inflation)
    alt_mean=mean.copy(); alt_mean[0]+=.5; alt_mean[1]+=.03
    code_a=np.array([pred.samples_pass(ch['kernel'],mean[None,:10])[0] for ch in choices])
    code_b=np.array([pred.samples_pass(ch['kernel'],alt_mean[None,:10])[0] for ch in choices])
    gaussian=dict(covariance_identical=bool(np.array_equal(cp,post_cov(cov,info,fixed.inflation))),
        mean_a=mean.tolist(),mean_b=alt_mean.tolist(),work_pass_vector_a=code_a.tolist(),work_pass_vector_b=code_b.tolist(),
        work_vector_changed=bool(np.any(code_a!=code_b)),synthetic_observations_only=True)
    suffix=diverse_order(bank,prior_mean,prior_cov,fixed.inflation)
    targets=[selectors[m].get('chosen') for m in ('fixed_prior_plan','plan_after_b0','adaptive_work_decision')]
    targets=[a for a in targets if a is not None]
    extra=next((a for a in suffix if a not in targets and any(c['base_id']==a for c in admitted)),None)
    if extra: targets.append(extra)
    targets=list(dict.fromkeys(targets)); branches=[]
    prefix_hash=plant.cache_identity()
    for repeat in range(4):
        cloned=plant.clone(); assert cloned.cache_identity()==prefix_hash
        # ONLY future noise is renewed; persistent bias and current AR state stay.
        cloned._rng=np.random.default_rng(stream(case,repeat,'source_branch_future'))
        prefix=dict(plant=cloned,estimator=est,block=b,physical_seconds=b.physical_seconds,history_kinds=kinds,suffix=suffix)
        for base_id in targets:
            result=run_policy(root,case,'fixed_prior_plan',5+repeat,1,0,
                tag=f'branch_continuations/{case["case_id"]}/{base_id}_{repeat}',prefix=prefix,forced=base_id)
            path=root/'source_branches'/f'{case["case_id"]}_{base_id}_{repeat}.json'; write_json(path,result)
            branches.append(dict(case_id=case['case_id'],object_id=case['object_id'],context=case['context'],candidate=base_id,repeat=repeat,
                result_path=str(path),normal_correct=result['normal_correct'],false_release=result['false_release'],
                remaining_physical_seconds=result['physical_seconds']-b.physical_seconds,serial_seconds=result['serial_seconds'],
                mass_error_kg=result['mass_error_kg'],com_error_m=result['com_error_m'],
                audit_pass_count=result['audit_pass_count'],violations=result['identification_violations']+result['verification_violations'],
                actual_first_action=result['motion_order'][0] if result['motion_order'] else None))
            immediate=next((s for s in result['steps'] if s['step']=='forced_source_branch'),{})
            branches[-1].update({k:immediate.get(k) for k in ('covariance_trace_gain','immediate_mass_error_kg','immediate_com_error_m')})
    return dict(case_id=case['case_id'],context=case['context'],group=case['group'],post_B0_estimate=est.estimate(),
        would_release_after_B0=release is not None,source_counterfactual_only=True,initial_checks=initial_checks,
        screens=screens,selectors=selectors,rankings=rankings,rank_check=rank_check,
        gaussian_value_test=gaussian,branches=branches,complete_prefix_clone_hash=prefix_hash,
        frozen_reference_suffix=suffix,screen_timeout=timed_out)
