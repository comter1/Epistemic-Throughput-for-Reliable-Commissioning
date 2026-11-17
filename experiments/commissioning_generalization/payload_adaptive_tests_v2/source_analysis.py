"""Source-only score ablations with the admitted set and deployment set held fixed."""
import json
from pathlib import Path
import numpy as np
from .cases import policy_view
from .evaluate import setup
from .prediction import Clock,DecisionTimeout
from .shared_decision import deployment_choices
from .policies import Selector
from .fixed_plans import diverse_order

def execute(root,case):
    root=Path(root); row=json.loads((root/'jobs/branches'/(case['case_id']+'.json')).read_text())['result']
    pub=policy_view(case); ctx,est,pred,bank,_=setup(root,pub,'adaptive_work_decision')
    prior_mean,prior_cov,_=est.posterior()
    mean=np.asarray(row['post_B0_estimate']['joint_mean17']); cov=np.asarray(row['post_B0_estimate']['joint_covariance17'])
    with np.load(root/'raw/branches'/case['case_id']/'prefix/public.npz') as z: start=z['q'][-1]
    choices,_,_=deployment_choices(pred,pub['configured_payload'],row['post_B0_estimate']['payload_estimate'],start,mean,cov,Clock(10),False,True)
    admitted=[]
    for screen in row['screens']:
        if screen['admitted']:
            a=bank['actions'][f'{screen["base_id"]}@{screen["speed"]}']
            admitted.append(dict(a['action'],information17=a['information17']))
    variants=[]
    for name,m,c in [('actual',mean,cov),('freeze_mean_to_prior',prior_mean,cov),('freeze_covariance_to_prior',mean,prior_cov)]:
        selector=Selector('adaptive_work_decision',bank,pub['configured_payload'],est.parameters['posterior_covariance_inflation'])
        selector.fallback_order=diverse_order(bank,mean,cov,selector.inflation)
        try:
            a,scores,fallback=selector.choose(m,c,admitted,pred,choices,Clock(10))
            variants.append(dict(variant=name,selected=a['base_id'] if a else None,scores=scores,fallback=fallback))
        except DecisionTimeout: variants.append(dict(variant=name,selected=None,reason='timeout'))
    return dict(case_id=case['case_id'],context=case['context'],group=case['group'],variants=variants,
                fixed_admitted_set=[a['base_id'] for a in admitted],deployment_configuration_set_frozen=True,
                interpretation='Model score intervention at one acquired history; no claim about actual performance gain.')
