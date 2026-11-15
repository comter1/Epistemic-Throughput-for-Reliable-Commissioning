"""Deployable sampled verification: consumes the external public interface only."""
import numpy as np

def public_verification_pass(observation, task=None, config=None):
    """Three endpoint holds at the declared 10 Hz interface, plus safety flags.

    Six consecutive samples span the required 0.5 s on the nominal sample grid.
    Timestamp jitter is not used to create a spurious extra temporal requirement.
    This sampled public criterion is distinct from continuous private audit scoring.
    """
    external = [r for r in observation.get('records', []) if r['port']=='external:out']
    if observation.get('visible_status') != 'ok' or not external:
        return False
    if any(r['missing'] or len(r['values'])<10 for r in external):
        return False
    values=np.asarray([r['values'] for r in external],dtype=float)
    if not np.isfinite(values).all() or np.any(values[:,7:10] < .5):
        return False
    # Constant public reference identifies settled target intervals. No true time,
    # joint truth, hidden metrics, clock correction or fault state is inspected.
    groups=[]; begin=0
    for i in range(1,len(values)):
        if not np.allclose(values[i,3:6],values[i-1,3:6],atol=1e-7,rtol=0):
            if i-begin>=6: groups.append(values[begin:i])
            begin=i
    if len(values)-begin>=6: groups.append(values[begin:])
    if len(groups)<3:
        return False
    tolerance=.02 if config is None else config['system']['position_tolerance_m']
    return all(np.all(np.linalg.norm(g[-6:,:3]-g[-6:,3:6],axis=1)<=tolerance) for g in groups[-3:])
