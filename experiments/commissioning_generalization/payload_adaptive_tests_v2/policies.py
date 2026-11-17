"""Next-pattern selection; no evaluator, label, or plant reference is accepted."""
import numpy as np
from payload_change_v1.design import _project_sample_matrix
from .fixed_plans import post_cov,diverse_order,beam
from .prediction import sqrt_psd,DecisionTimeout

def entropy(codes):
    counts=np.bincount(codes,minlength=16); p=counts[counts>0]/len(codes)
    return float(-np.sum(p*np.log2(p)))

class Selector:
    def __init__(self,method,bank,geometry,inflation,fixed_template=0):
        self.method=method; self.bank=bank; self.geometry=geometry; self.inflation=inflation
        self.template=fixed_template; self.order=None; self.fallback_order=None; self.plan_trace=[]

    def score(self,mean,cov,candidates,predictor,choices,clock,kind=None,samples=128,outcomes=16):
        kind=kind or self.method; Y=np.asarray(self.bank['work_Y']).reshape(-1,10)
        priorvar=float(np.mean(np.einsum('ni,ij,nj->n',Y,cov[:10,:10],Y)))
        # Same draws for all candidate designs; all 17 coordinates retain bias correlation.
        rng=np.random.default_rng(413819); zs=rng.standard_normal((samples,10)); zo=rng.standard_normal((outcomes,17))
        def states(s):
            codes=np.zeros(len(s),np.int32)
            for j,ch in enumerate(choices): codes|=predictor.samples_pass(ch['kernel'],s,clock).astype(np.int32)<<j
            return codes
        if kind=='decision':
            sampled=_project_sample_matrix(mean[:10]+zs@sqrt_psd(cov[:10,:10]).T,self.geometry)
            before=entropy(states(sampled))
        else: before=0.
        rows=[]
        for a in candidates:
            clock.check(); cp=post_cov(cov,np.asarray(a['information17']),self.inflation)
            aftervar=float(np.mean(np.einsum('ni,ij,nj->n',Y,cp[:10,:10],Y)))
            vargain=(priorvar-aftervar)/a['duration']; score=vargain; variability=0.; after=before
            if kind=='decision':
                mus=mean+zo@sqrt_psd(cov-cp).T
                draws=_project_sample_matrix((mus[:,None,:10]+(zs@sqrt_psd(cp[:10,:10]).T)[None,:,:]).reshape(-1,10),self.geometry)
                codes=states(draws).reshape(outcomes,samples)
                ent=np.array([entropy(c) for c in codes]); after=float(ent.mean())
                score=(before-after)/a['duration']; variability=float(ent.std(ddof=1)/np.sqrt(outcomes)/a['duration'])
            rows.append(dict(base_id=a['base_id'],speed=a['speed'],duration=a['duration'],score=float(score),
                mc_standard_error=variability,entropy_before=before,expected_entropy_after=after,
                variance_gain_per_s=vargain,score_kind=kind,feasible=True))
        return rows

    def freeze(self,mean,cov,candidates,predictor,choices,clock,after_b0=False):
        diverse=diverse_order(self.bank,mean,cov,self.inflation); self.fallback_order=diverse
        if self.method=='fixed_prior_plan' and self.template==0:
            self.order=diverse[:4]+[a for a in diverse if a not in diverse[:4]]; return
        kind='variance' if self.method=='fixed_prior_plan' and self.template==1 else 'decision'
        # A context-only kinematic omission is permanent; uncertain omissions are
        # appended and reconsidered by the live shared screen, never deleted.
        try:
            seq,traces=beam(mean,cov,candidates,lambda m,c,a,k:self.score(m,c,a,predictor,choices,k,kind=kind),self.inflation,clock)
            self.plan_trace=traces; self.order=seq+[a for a in diverse if a not in seq]
        except DecisionTimeout:
            self.order=diverse; self.plan_trace=[dict(fallback_reason='bounded_search_timeout',actual_seconds=clock.used)]

    def choose(self,mean,cov,candidates,predictor,choices,clock):
        if not self.fallback_order: self.fallback_order=diverse_order(self.bank,mean,cov,self.inflation)
        if not candidates: return None,[], 'no_motion_admitted'
        if self.method=='b0_verify_only': return None,[],'initial_only'
        if self.method in ('fixed_prior_plan','plan_after_b0'):
            if self.order is None: self.freeze(mean,cov,candidates,predictor,choices,clock,True)
            picked=min(candidates,key=lambda a:self.order.index(a['base_id']))
            return picked,[],None
        if self.method=='simple_parameter_rule':
            y=np.asarray(self.bank['work_Y']).reshape(-1,10)
            static=float(np.mean(np.einsum('ni,ij,nj->n',y[:,:4],cov[:4,:4],y[:,:4])))
            dynamic=float(np.mean(np.einsum('ni,ij,nj->n',y[:,4:],cov[4:10,4:10],y[:,4:])))
            prefix='S' if static>=dynamic else 'D'
            ranked=sorted(candidates,key=lambda a:(a['base_id'][0]!=prefix,self.fallback_order.index(a['base_id'])))
            return ranked[0],[dict(base_id=a['base_id'],score=float(a['base_id'][0]==prefix),
                static_work_torque_variance=static,inertia_work_torque_variance=dynamic,score_kind='parameter_rule') for a in candidates],None
        kind='variance' if self.method=='adaptive_variance' else 'decision'
        scores=self.score(mean,cov,candidates,predictor,choices,clock,kind=kind)
        if kind=='decision' and (max(r['score'] for r in scores)-min(r['score'] for r in scores)<1e-7 or max(r['score'] for r in scores)<=0):
            return min(candidates,key=lambda a:self.fallback_order.index(a['base_id'])),scores,'decision_scores_tied_or_nonpositive_diverse_fallback'
        selected=min(scores,key=lambda r:(-r['score'],r['base_id']))['base_id']
        return next(a for a in candidates if a['base_id']==selected),scores,None
