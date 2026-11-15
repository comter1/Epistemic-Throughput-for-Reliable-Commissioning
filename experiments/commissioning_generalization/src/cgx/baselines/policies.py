"""Common-information acquisition rules and frozen learned policy wrappers."""
from __future__ import annotations
import copy
import math
import time
from collections import defaultdict
import numpy as np
import torch
from cgx.models.features import action_dict, RESOURCES


def affordable(action,remaining):
    a=action_dict(action)
    return a['kind']=='stop' or all(a['cost_bound'][r]<=remaining.get(r,0) for r in RESOURCES)


def _stop(actions):
    return next(a for a in actions if action_dict(a)['kind']=='stop')


def _measured_suspects(history):
    """Compare acquired paired values, stale IDs and clock-adjusted delay evidence.

    No threshold is selected from test data. Nominal adapter gain is public.
    Scores are continuous, reducing arbitrary graph-order ties.
    """
    g=history['graph']; nodes={n['id']:n for n in g['nodes']}
    incoming=defaultdict(list)
    for e in g['edges']:incoming[e['target']].append(e['source'])
    representation={};input_representation={};pending=set(nodes)
    while pending:
        ready=[i for i in pending if all(j in representation for j in incoming[i])]
        if not ready:raise ValueError('Public graph is cyclic')
        for i in ready:
            n=nodes[i];upstream=incoming[i]
            input_representation[i]=representation[upstream[0]] if upstream else (1.,0.)
            if n['type']=='adapter':representation[i]=(n['nominal'].get('scale',1.),n['nominal'].get('offset',0.))
            elif n['type'] in ('sensor','controller','merger'):representation[i]=(1.,0.)
            else:representation[i]=input_representation[i]
            pending.remove(i)
    score=defaultdict(float); seen=defaultdict(int); calibrated=set(); offsets={}
    for obs in history.get('observations',()):
        for burst in obs.get('calibration_bursts',()):
            a,b=burst['pair']; t=np.asarray(burst['timestamps_s'])
            value=np.mean(((t[:,1]-t[:,0])+(t[:,2]-t[:,3]))/2)
            offsets[(a,b)]=value; offsets[(b,a)]=-value; calibrated.add(frozenset((a,b)))
        grouped=defaultdict(list)
        for r in obs.get('records',()): grouped[r['port']].append(r)
        for edge in g['edges']:
            source,target=edge['source'],edge['target']
            left=grouped.get(source+':'+edge['source_port'],[])
            right=grouped.get(target+':'+edge['target_port'],[])
            c1,c2=nodes[source]['clock_domain'],nodes[target]['clock_domain']
            if c1!=c2 and (c1,c2) not in offsets: continue
            correction=offsets.get((c1,c2),0.)
            bytoken={r['message_token']:r for r in left}
            delays=[r['local_time_s']-bytoken[r['message_token']]['local_time_s']-correction for r in right if r['message_token'] in bytoken]
            if delays:
                residual=abs(float(np.median(delays)))/.01
                score[source]+=residual/2; score[target]+=residual/2
        for n in g['nodes']:
            ident=n['id']; ins=grouped.get(ident+':in',[]); outs=grouped.get(ident+':out',[])
            if ins or outs: seen[ident]+=1
            if n['type']=='cache' and outs:
                repeats=np.mean([a['message_token']==b['message_token'] for a,b in zip(outs[1:],outs[:-1])]) if len(outs)>1 else 0
                score[ident]+=repeats
            if not ins or not outs: continue
            byid=defaultdict(list)
            for r in ins: byid[r['message_token']].append(r)
            for out in outs:
                if out['message_token'] not in byid: continue
                inp=min(byid[out['message_token']],key=lambda r:abs(r['local_time_s']-out['local_time_s']))
                dt=out['local_time_s']-inp['local_time_s']
                if n['type'] in ('queue','cache'): score[ident]+=abs(dt)/(len(outs)*.01)
                if n['type']=='adapter':
                    before=np.asarray(inp['values']); after=np.asarray(out['values'])
                    if len(before)==len(after) and len(before):
                        # Nominal output representation is declared; input is inferred
                        # from the acquired upstream interface contract, never truth.
                        nom=n['nominal']; scale=nom.get('scale',1.); shift=nom.get('offset',0.)
                        input_scale,input_shift=input_representation[ident]
                        score[ident]+=float(np.linalg.norm(after-((before-input_shift)/input_scale*scale+shift)))/(len(outs)*max(.001,float(np.linalg.norm(before))))
    return score,seen,calibrated


def interface_rule(history,actions,prediction=None):
    actions=[a for a in actions if affordable(a,history['remaining'])]
    suspect,seen,calibrated=_measured_suspects(history)
    nodes={n['id']:n for n in history['graph']['nodes']}
    probabilities={}
    if prediction:
        probabilities=dict(zip(prediction['target_keys'],np.exp2(prediction['log_probabilities_base2'])))
    choices=[]
    for original in actions:
        a=action_dict(original)
        if a['kind']=='stop': choices.append((0.,original)); continue
        ids=set(p.split(':')[0] for p in a['ports']); ids.intersection_update(nodes)
        plausibility=sum(sum(p for k,p in probabilities.items() if k.startswith(i+':')) for i in ids)
        evidence=sum(suspect[i] for i in ids)
        novelty=sum(1/(1+seen[i]) for i in ids)
        pair=a.get('calibration_pair'); new_cal=bool(pair) and frozenset(pair) not in calibrated
        if a['kind']=='calibration_only':
            # A calibration is useful once corresponding cross-clock observations exist.
            observed_domains={nodes[i]['clock_domain'] for i in seen if seen[i]}
            value=1.5 if new_cal and set(pair)<=observed_domains else .01
        else:
            value=novelty+2*plausibility+min(5,evidence)+.4*(a['kind'] in ('module_pair','link_pair','pair_plus_calibration'))
            value+=.75*new_cal; value*=.5+.5*a['sample_hz']/100
            value/=1+.5*a['cost_bound']['bytes']/max(1,history['remaining']['bytes'])
            if a['kind']=='full_trace': value/=max(1,math.sqrt(len(ids)))
        choices.append((value,original))
    return max(choices,key=lambda item:item[0])[1]


class RuntimePolicy:
    """No parameter or normalization updates occur in choose/plan."""
    def __init__(self,method,model=None,beta=0.5):
        self.method=method; self.model=model; self.beta=beta
        self.fixed_plan=method in ('gnn_fixed_plan','full_recording_feasible','full_recording_unrestricted_reference')
        if model is not None: model.eval()
    def choose(self,history,actions,prediction=None):
        actions=[a for a in actions if affordable(a,history['remaining'])]
        if self.method=='default_only': return _stop(actions)
        if self.method=='interface_rule': return interface_rule(history,actions,prediction)
        if self.method.startswith('full_recording'):
            full=[a for a in actions if action_dict(a)['kind']=='full_trace']
            if full: return max(full,key=lambda a:action_dict(a)['sample_hz'])
            cal=[a for a in actions if action_dict(a)['kind']=='calibration_only']
            if cal: return interface_rule(history,cal+[_stop(actions)],prediction)
            return _stop(actions)
        if self.model is None: raise RuntimeError('Learned policy requires a frozen selected checkpoint')
        state=copy.deepcopy(history)
        if self.method=='gnn_depth2': state['remaining']['decisions']=min(2,state['remaining']['decisions'])
        if self.method=='gnn_myopic': state['remaining']['decisions']=min(1,state['remaining']['decisions'])
        with torch.inference_mode():
            if self.method=='cost_aware_adaptation':
                values,loss=self.model.score(state,actions,prediction,return_next_loss=True)
                losses=loss.cpu().numpy(); cost=np.asarray([sum(action_dict(a)['cost_bound'][k]/max(1,history['remaining'][k]) for k in ('bytes','executions','calibrations')) for a in actions])
                norm=lambda x:(x-x.min())/max(1e-8,x.max()-x.min())
                for j,action in enumerate(actions):
                    if action_dict(action)['kind']=='stop':
                        if prediction:
                            lp=np.asarray(prediction['log_probabilities_base2']); losses[j]=-float(np.sum(np.exp2(lp)*lp))
                        else: losses[j]=0.
                score=-(self.beta*norm(losses)+(1-self.beta)*norm(cost))
                return actions[int(np.argmax(score))]
            scores=self.model.score(state,actions,prediction)
            if not torch.isfinite(scores).all(): raise FloatingPointError('Nonfinite acquisition score')
            return actions[int(scores.argmax().item())]
    def plan(self,initial,actions,prediction=None):
        """Plan completely before returning any new evidence; budgets use upper bounds."""
        from cgx.evaluation.rollout import call_deadline
        h=copy.deepcopy(initial); plan=[]; h['planned_actions']=[]; decision_times=[]
        while h['remaining']['decisions']>0:
            stamp=time.monotonic()
            with call_deadline(2.0):
                action=self.choose(h,actions,prediction)
            elapsed=time.monotonic()-stamp;decision_times.append(elapsed)
            if elapsed>2.0:raise TimeoutError('Fixed-plan choice exceeded2s decision limit')
            a=action_dict(action)
            if a['kind']=='stop': break
            plan.append(action); h['planned_actions'].append(a)
            h['action_keys']=list(h.get('action_keys',[]))+[a['action_key']]
            h['remaining']={k:h['remaining'][k]-a['cost_bound'][k] for k in RESOURCES}
        self.last_plan_decision_seconds=decision_times
        return plan
