"""Source-designed information matrices and bounded fixed-priority searches."""
from pathlib import Path
import json
import numpy as np
from payload_change_v1.estimator import default_regressor, whiten_features, SCALE
from .protocol import write_json
from .trajectories import grid,library,scaled,work

def build_banks(root):
    root=Path(root); pars=json.loads((root/'configs/estimator_parameters.json').read_text())
    contexts=json.loads((root/'configs/workstations.json').read_text()); reg=default_regressor()
    for name,ctx in contexts.items():
        actions={}
        for a in library():
            for speed in (1.,.75,.5):
                action=scaled(a,speed); g=grid(action,hz=20)
                known,y=reg.batch(g['q'],g['dq'],g['ddq'])
                h,_=whiten_features(dict(Y=y,y=np.zeros_like(known),time=g['time']),pars)
                actions[f'{a["id"]}@{speed}']=dict(action=action,information17=(h.T@h).tolist())
        g=grid(work(ctx),hz=20); _,y=reg.batch(g['q'],g['dq'],g['ddq'])
        write_json(root/'configs'/f'design_{name}.json',dict(actions=actions,work_Y=y.tolist()))

def post_cov(cov,info,inflation):
    scaled_cov=cov/np.outer(SCALE,SCALE)
    inv=np.linalg.inv(scaled_cov+np.eye(17)*1e-12)
    return np.linalg.inv(inv+info*np.outer(SCALE,SCALE)/inflation)*np.outer(SCALE,SCALE)

def diverse_order(bank,mean,cov,inflation):
    """Complementary static poses followed by inertia-diverse dynamic patterns."""
    chosen=[]; c=cov.copy()
    for prefix,interest in [('S',slice(0,4)),('D',slice(4,10))]:
        pool=[a['id'] for a in library() if a['id'].startswith(prefix)]
        while pool:
            scores={}
            for a in pool:
                cp=post_cov(c,np.asarray(bank['actions'][a+'@1.0']['information17']),inflation)
                sign,ld=np.linalg.slogdet(cp[interest,interest]/np.outer(SCALE[interest],SCALE[interest]))
                scores[a]=-ld if sign>0 else -1e30
            best=min(pool,key=lambda a:(-scores[a],a)); pool.remove(best); chosen.append(best)
            c=post_cov(c,np.asarray(bank['actions'][best+'@1.0']['information17']),inflation)
    return chosen

def beam(mean,cov,candidates,score_fn,inflation,clock):
    """Width three, at most four steps, zero-innovation prospective means."""
    frontier=[([],cov,0.)]; traces=[]; best=[]
    for depth in range(4):
        children=[]
        for seq,c,total in frontier:
            clock.check(); remaining=[a for a in candidates if a['base_id'] not in seq]
            scores=score_fn(mean,c,remaining,clock)
            for row in scores:
                a=next(x for x in remaining if x['base_id']==row['base_id'])
                cp=post_cov(c,np.asarray(a['information17']),inflation)
                children.append((seq+[a['base_id']],cp,total+row['score']*a['duration']))
            traces.append(dict(depth=depth,prefix=seq,scores=scores))
        if not children: break
        frontier=sorted(children,key=lambda x:(-x[2],x[0]))[:3]
        best=frontier[0][0]
    return best,traces
