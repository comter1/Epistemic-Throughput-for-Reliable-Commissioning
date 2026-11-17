"""Complete simulated commissioning, private audits, and evaluator references."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
import numpy as np
from cgx.env.system import HOME
from payload_change_v1.estimator import PayloadEvidenceEstimator,payload_to_pi,SCALE
from .protocol import write_json,digest,REQUIREMENTS
from .worker import context,new_plant,save_block
from .trajectories import b0,work
from .workstations import kinematic
from .prediction import PublicPredictionModel,Clock,DecisionTimeout
from .motion_screen import candidates,class_alarms
from .shared_decision import deployment_choices,public_verification_pass,can_update_at_rest,serialize_choice
from .policies import Selector

def setup(root,public,method,multiplier=1,template=0):
    root=Path(root); ctx=json.loads((root/'configs/workstations.json').read_text())[public['context']]
    pars=json.loads((root/'configs/estimator_parameters.json').read_text())
    pars['posterior_covariance_inflation']*=multiplier
    margins=json.loads((root/'configs/prediction_margins.json').read_text())
    predictor=PublicPredictionModel(root,ctx,public['configured_payload'],margins)
    bank=json.loads((root/'configs'/f'design_{public["context"]}.json').read_text())
    estimator=PayloadEvidenceEstimator(pars,geometry=public['configured_payload'])
    selector=Selector(method,bank,public['configured_payload'],pars['posterior_covariance_inflation'],template)
    return ctx,estimator,predictor,bank,selector

def initial_motion(predictor,public,clock):
    """Conservative bounding-box ellipsoid covers the complete initial payload set."""
    cfg=public['configured_payload']; center=np.asarray(public['box_center']); half=np.asarray(public['box_size'])/2
    mean=np.zeros(17); mean[:10]=payload_to_pi(dict(cfg,mass=.85,com=center.tolist()))
    # Bounds deliberately cover independent moment extremes of any object in the known box.
    lo=np.r_[.2, -1.5*(abs(center)+half),np.zeros(3),[-.03]*3]
    hi=np.r_[1.5, 1.5*(abs(center)+half),[.06]*3,[.03]*3]
    width=np.maximum(abs(lo-mean[:10]),abs(hi-mean[:10]))
    cov=np.zeros((17,17)); cov[:10,:10]=np.diag(width**2*10)
    checks=[]
    for sign in (1,-1,0):
        a=b0(sign); geometry=kinematic(cfg,predictor.context,a)
        k=predictor.kernel(cfg,a,HOME,clock); pred=predictor.summary(k,mean,cov,radius=1.)
        # Linear-region status is reported separately. This initial engineering
        # range screen has no empirical residual alarm before acquiring evidence.
        okay=geometry['admissible'] and not pred['unsupported'] and pred['clearance_m']>0 and np.all(np.asarray(pred['torque_Nm'])<np.array([87]*4+[12]*3))
        checks.append(dict(sign=sign,geometry=geometry,prediction=pred,pass_screen=bool(okay)))
        if okay or sign==0: return a,checks

def offline_candidates(bank):
    return [dict(v['action'],information17=v['information17']) for k,v in bank['actions'].items() if k.endswith('@1.0')]

def summarize(result,case,method,seed,multiplier,template,steps,blocks,audits,estimator,
              physical,verify_time,compute,updates,reason,status,final_speed):
    estimate=estimator.estimate(); true=payload_to_pi(case['true_payload']); mean,cov,_=estimator.posterior()
    delta=true-mean[:10]; sigma=np.sqrt(np.maximum(np.diag(cov[:10,:10]),1e-30))
    audits_pass=sum(a['metrics']['pass'] for a in audits); released=status in ('approved_normal','approved_half')
    violations=lambda records:sum(bool(r['metrics']['hard_abort'] or r['metrics']['unexpected_contact'] or r['metrics']['joint_limit_violation']
                           or r['metrics']['saturation_fraction']>.01+1e-12 or r['metrics']['max_continuous_saturation_seconds']>.1+1e-12) for r in records)
    idblocks=[b for b in blocks if b['kind']=='identification']; vblocks=[b for b in blocks if b['kind']=='verification']
    return dict(case_id=case['case_id'],object_id=case['object_id'],family_id=case['family_id'],context=case['context'],group=case['group'],
        method=method,noise_seed=seed,multiplier=multiplier,fixed_template=template,status=status,release=released,
        normal_correct=bool(status=='approved_normal' and audits_pass==8),half_correct=bool(status=='approved_half' and audits_pass==8),
        false_release=bool(released and audits_pass<8),audit_pass_count=audits_pass,audit_denominator=8 if released else 0,
        identification_seconds=physical,verification_seconds=verify_time,physical_seconds=physical+verify_time,
        serial_seconds=physical+verify_time+sum(compute.values()),compute=compute,updates=updates,stop_reason=reason,final_speed=final_speed,
        capped_serial_seconds=physical+verify_time+sum(compute.values()) if status=='approved_normal' and audits_pass==8 else 160.,
        identification_violations=violations(idblocks),verification_violations=violations(vblocks),audit_violations=violations(audits),
        identification_tracking_failures=sum(not b['metrics']['pass'] for b in idblocks),
        mass_error_kg=abs(float(delta[0])),com_error_m=float(np.linalg.norm(np.asarray(case['true_payload']['com'])-np.asarray(estimate['payload_estimate']['com']))),
        mass_covered=bool(abs(delta[0])<=2.5758293035489004*sigma[0]),
        com_covered=bool(np.all(abs(np.asarray(case['true_payload']['com'])-np.asarray(estimate['payload_estimate']['com']))<=2.5758293035489004*np.asarray(estimate['com_sd']))),
        joint_covered=bool((delta/SCALE[:10])@np.linalg.solve(cov[:10,:10]/np.outer(SCALE[:10],SCALE[:10])+np.eye(10)*1e-12,delta/SCALE[:10])<=23.209251158954356),
        estimate=estimate,steps=steps,blocks=blocks,audits=audits,
        motion_order=[b['action']['id'] for b in idblocks],motion_speeds=[b['action']['speed'] for b in idblocks],
        repeat_count=len(idblocks)-len(set(b['action']['id'] for b in idblocks)))

def run_policy(root,case,method,seed,multiplier=1,template=0,tag='evaluation',prefix=None,forced=None):
    from .cases import policy_view
    root=Path(root); public=policy_view(case); ctx,est,pred,bank,selector=setup(root,public,method,multiplier,template)
    key=f'{case["case_id"]}_{seed}_{method}_m{multiplier}_t{template}'
    base=Path('raw')/tag/key
    plant=new_plant(root,case,seed); steps=[]; blocks=[]; audits=[]; history_kinds={}; used=[]
    physical=0.; verify_time=0.; compute=dict(estimator=0.,prediction_planning=0.,configuration_update=0.)
    compute_wall=dict(estimator=0.,prediction_planning=0.,configuration_update=0.)
    updates=[]; reason='identification_budget'; status='unresolved'; final_speed=None
    def execute(a,kind,index):
        nonlocal physical,verify_time
        b=save_block(root,plant,a,base/f'{kind}_{index:02d}',tag+'_'+kind)
        blocks.append(dict(action=a,kind=kind,metrics=b.private['metrics'],public_metrics=b.private['public_metrics'],
                           physical_seconds=b.physical_seconds,path=str(base/f'{kind}_{index:02d}')))
        if kind=='identification': physical+=b.physical_seconds
        else: verify_time+=b.physical_seconds
        t=time.process_time();wall=time.perf_counter(); est.ingest(b.public|{'execution_id':b.metadata['execution_id']}); compute['estimator']+=time.process_time()-t
        compute_wall['estimator']+=time.perf_counter()-wall
        history_kinds[b.metadata['execution_id']]=a['kind']
        return b
    if prefix is None:
        clock=Clock(10.)
        try:
            initial,initial_checks=initial_motion(pred,public,clock)
        except DecisionTimeout:
            initial=b0(0); initial_checks=[dict(reason='initial_screen_timeout_hold')]
        if method=='fixed_prior_plan':
            try:
                mean,cov,_=est.posterior(); estimate=est.estimate()['payload_estimate']
                choices,_,_=deployment_choices(pred,public['configured_payload'],estimate,HOME,mean,cov,clock,False,True)
                selector.freeze(mean,cov,offline_candidates(bank),pred,choices,clock)
            except DecisionTimeout:
                from .fixed_plans import diverse_order
                mean,cov,_=est.posterior(); selector.order=diverse_order(bank,mean,cov,selector.inflation)
                selector.plan_trace=[dict(fallback_reason='prior_plan_timeout_initial_motion_unchanged')]
        compute['prediction_planning']+=clock.used
        compute_wall['prediction_planning']+=clock.wall_used
        block=execute(initial,'identification',0)
        steps.append(dict(step=-1,initial_screen=initial_checks,priority=selector.order,plan_trace=selector.plan_trace))
    else:
        plant=prefix['plant'].clone(); est=copy.deepcopy(prefix['estimator']); physical=prefix['physical_seconds']
        block=prefix['block']; history_kinds=copy.deepcopy(prefix['history_kinds'])
        selector.order=list(prefix['suffix']); selector.fallback_order=list(prefix['suffix'])
        if forced is not None:
            clock=Clock(10); mean,cov,_=est.posterior()
            settings=est.estimate()['payload_estimate']
            eligible,screen,_=candidates(pred,bank,settings,block.public['q'][-1],mean,cov,60-physical,class_alarms(est,history_kinds),clock)
            chosen=next((a for a in eligible if a['base_id']==forced),None)
            if chosen:
                plant.configure(chosen['configured'])
                before_mean,before_cov,_=est.posterior()
                block=execute({k:v for k,v in chosen.items() if k not in ('information17','prediction','configured')},'identification',1)
                used.append(chosen['base_id'])
                after_mean,after_cov,_=est.posterior()
                steps.append(dict(step='forced_source_branch',chosen_action=chosen['base_id'],
                    before_mean17=before_mean.tolist(),after_mean17=after_mean.tolist(),
                    covariance_trace_gain=float(np.trace(before_cov[:10,:10]-after_cov[:10,:10])),
                    immediate_mass_error_kg=abs(float(after_mean[0]-case['true_payload']['mass'])),
                    immediate_com_error_m=float(np.linalg.norm(after_mean[1:4]/after_mean[0]-np.asarray(case['true_payload']['com'])))))
            compute['prediction_planning']+=clock.used
            compute_wall['prediction_planning']+=clock.wall_used
    for index in range(7-len(used)):
        if block.private['metrics']['hard_abort']:
            status='aborted'; reason='physical_watchdog'; break
        if sum(compute.values())>=80:
            reason='online_compute_budget'; break
        clock=Clock(min(10.,80-sum(compute.values())))
        mean,cov,_=est.posterior(); estimate=est.estimate(); start=block.public['q'][-1]
        record=dict(step=index,mean17=mean.tolist(),covariance17=cov.tolist(),residual_consistent=estimate['residual_consistent'],
                    remaining_identification_s=60-physical,at_rest=can_update_at_rest(block.public))
        chosen=None; eligible_work=None; choices=[]; score_rows=[]
        try:
            choices,eligible_work,errors=deployment_choices(pred,public['configured_payload'],estimate['payload_estimate'],start,mean,cov,clock,estimate['residual_consistent'])
            record.update(deployment_choices=[serialize_choice(c) for c in choices],prediction_errors=errors)
            if eligible_work is None and method!='b0_verify_only' and len(used)<6:
                next_config=estimate['payload_estimate'] if record['at_rest'] else plant.configured_payload
                eligible,screen,timed_out=candidates(pred,bank,next_config,start,mean,cov,60-physical,class_alarms(est,history_kinds),clock)
                record['motion_screen']=screen
                # Fixed relative priorities are immutable, but previously acquired
                # patterns are traversed once before any repeat cycle.
                available=eligible
                if method in ('fixed_prior_plan','plan_after_b0','simple_parameter_rule'):
                    fresh=[a for a in eligible if a['base_id'] not in used]
                    if fresh: available=fresh
                if timed_out:
                    record['fallback_reason']='motion_screen_time_budget'
                    if available:
                        from .fixed_plans import diverse_order
                        order=selector.order or diverse_order(bank,mean,cov,selector.inflation)
                        chosen=min(available,key=lambda a:order.index(a['base_id']))
                else:
                    chosen,score_rows,fallback=selector.choose(mean,cov,available,pred,choices,clock)
                    record['fallback_reason']=fallback
                record.update(candidate_scores=score_rows,priority=selector.order,plan_trace=selector.plan_trace)
        except DecisionTimeout:
            record['fallback_reason']='prediction_or_selection_timeout_unresolved'
        compute['prediction_planning']+=clock.used
        compute_wall['prediction_planning']+=clock.wall_used
        record['decision_cpu_seconds']=clock.used
        record['decision_wall_seconds']=clock.wall_used
        record['public_prediction_cache_hits']=pred.cache_hits
        if eligible_work is not None:
            if not record['at_rest']:
                reason='setting_change_requires_rest'; record['stop_reason']=reason; steps.append(record); break
            config=eligible_work['configured']; t=time.process_time();wall=time.perf_counter()
            if digest(config)!=digest(plant.configured_payload):
                updates.append(dict(at_physical_seconds=physical,from_config=plant.configured_payload,to_config=config,reason='eligible_work_choice'))
                plant.configure(config)
            compute['configuration_update']+=time.process_time()-t
            compute_wall['configuration_update']+=time.perf_counter()-wall
            record['chosen_work']=serialize_choice(eligible_work); steps.append(record)
            final_speed=eligible_work['speed']; verification=execute(eligible_work['action'],'verification',0)
            if public_verification_pass(verification):
                status='approved_normal' if final_speed==1. else 'approved_half'; reason='prediction_and_public_verification_pass'
                for j in range(8):
                    # Independent evaluator-only jobs, each with a complete paid
                    # approach from rest. No audit data flows back to the policy.
                    audit_plant=new_plant(root,case,seed,configured=config,tag=f'audit_{j}')
                    variation=int(digest(['work_variation',case['context'],seed,j])[:8],16)
                    a=work(ctx,final_speed,variation)
                    ab=save_block(root,audit_plant,a,base/f'audit_{j:02d}',tag+'_private_audit')
                    audits.append(dict(index=j,action=a,metrics=ab.private['metrics'],path=str(base/f'audit_{j:02d}')))
            else:
                status='aborted' if verification.private['metrics']['hard_abort'] else 'unresolved'
                reason='single_public_verification_failed'
            break
        if chosen is None:
            reason='initial_only_no_eligible_work' if method=='b0_verify_only' else ('six_additional_blocks' if index==6 else 'no_admitted_action_or_prediction_timeout')
            record['stop_reason']=reason; steps.append(record); break
        if not record['at_rest'] and digest(chosen['configured'])!=digest(plant.configured_payload):
            reason='update_requires_rest'; steps.append(record); break
        t=time.process_time();wall=time.perf_counter()
        if digest(chosen['configured'])!=digest(plant.configured_payload):
            updates.append(dict(at_physical_seconds=physical,from_config=plant.configured_payload,to_config=chosen['configured'],reason='next_motion_shared_screen'))
            plant.configure(chosen['configured'])
        compute['configuration_update']+=time.process_time()-t
        compute_wall['configuration_update']+=time.perf_counter()-wall
        action={k:v for k,v in chosen.items() if k not in ('information17','prediction','configured')}
        record['chosen_action']=action; steps.append(record); used.append(action['base_id'])
        block=execute(action,'identification',index+1)
    result=summarize({},case,method,seed,multiplier,template,steps,blocks,audits,est,physical,verify_time,compute,updates,reason,status,final_speed)
    result.update(compute_wall=compute_wall,observed_wall_based_serial_seconds=physical+verify_time+sum(compute_wall.values()))
    return result

def reference_job(root,case,seed):
    ctx,_=context(root,case); records=[]
    for name,cfg,speed in [('initial_normal',case['configured_payload'],1.),('true_normal',case['true_payload'],1.),('true_half',case['true_payload'],.5)]:
        p=new_plant(root,case,seed,cfg,tag='reference_'+name)
        path=Path('raw/reference')/f'{case["case_id"]}_{seed}'/name
        b=save_block(root,p,work(ctx,speed),path,'reference_'+name)
        records.append(dict(condition=name,metrics=b.private['metrics'],physical_seconds=b.physical_seconds,path=str(path)))
    passed={r['condition']:r['metrics']['pass'] for r in records}
    category=('initial_settings_pass' if passed['initial_normal'] else 'needs_update_normal' if passed['true_normal'] else 'needs_half_speed' if passed['true_half'] else 'neither_declared_reference_passes')
    audits=[]
    speed=1. if passed['true_normal'] else .5
    if passed['true_normal'] or passed['true_half']:
        for j in range(8):
            p=new_plant(root,case,seed,case['true_payload'],tag=f'audit_{j}')
            variation=int(digest(['work_variation',case['context'],seed,j])[:8],16)
            path=Path('raw/reference')/f'{case["case_id"]}_{seed}'/f'true_audit_{j}'
            b=save_block(root,p,work(ctx,speed,variation),path,'true_reference_audit')
            audits.append(dict(index=j,metrics=b.private['metrics'],path=str(path)))
    return dict(case_id=case['case_id'],object_id=case['object_id'],family_id=case['family_id'],context=case['context'],group=case['group'],noise_seed=seed,
                method='true_payload_reference',category=category,conditions=records,audits=audits,
                normal_correct=bool(passed['true_normal'] and len(audits)==8 and all(a['metrics']['pass'] for a in audits)))

def execute_job(root,job):
    if job['stage']=='reference': return reference_job(root,job['case'],job['seed'])
    return run_policy(root,job['case'],job['method'],job['seed'],job.get('multiplier',1),job.get('template',0),job['stage']+'/'+digest(job)[:12])
