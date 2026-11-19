"""Executable estimator checks; synthetic checks are never experiment outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from functools import lru_cache
import numpy as np

from .estimator import (PayloadRegressor, PayloadEvidenceEstimator, default_parameters,
                        payload_to_pi, pi_to_payload, pseudo_inertia, project_physical)
from .design import choose_action, _project_sample_matrix


def checks():
    from .physics import Dynamics
    from .trajectories import action_library
    from cgx.env.system import HOME
    start = time.perf_counter()
    rng = np.random.default_rng(4401)
    reg = PayloadRegressor()
    reg_half = PayloadRegressor(step_multiplier=.5)
    errors, steps = [], []
    for _ in range(12):
        m = rng.uniform(.25, 1.4)
        c = rng.uniform(-.025, .025, 3) + [0, 0, .07]
        sides = rng.uniform(.04, .10, 3)
        ic = m/12*np.diag([sides[1]**2+sides[2]**2, sides[0]**2+sides[2]**2, sides[0]**2+sides[1]**2])
        load = {"mass": m, "com": c, "inertia_com": ic}
        pi = payload_to_pi(load)
        q = HOME+rng.uniform(-.25, .25, 7)
        dq, ddq = rng.normal(0, .2, 7), rng.normal(0, .8, 7)
        known, y = reg.at(q, dq, ddq)
        kh, yh = reg_half.at(q, dq, ddq)
        predicted = known+y@pi
        truth = Dynamics(pi_to_payload(pi)).inverse(q,dq,ddq)
        errors.append(float(np.max(np.abs(predicted-truth))))
        steps.append(float(np.max(np.abs(y-yh))))
        restored = payload_to_pi(pi_to_payload(pi))
        assert np.max(np.abs(pi-restored)) < 1e-12
        assert np.linalg.eigvalsh(pseudo_inertia(pi))[0] > 0
    assert max(errors) < 1e-7, errors
    assert max(steps) < 1e-6, steps
    statics = []
    for action in action_library()[:6]:
        _, y = reg.at(action["pose"],np.zeros(7),np.zeros(7))
        statics.append(y)
    static = np.concatenate(statics)
    assert np.max(np.abs(static[:,4:])) < 1e-8
    rank_single = int(np.linalg.matrix_rank(statics[0][:,:4],tol=1e-6))
    rank_multiple = int(np.linalg.matrix_rank(static[:,:4],tol=1e-6))
    assert rank_multiple >= rank_single
    first_singular = np.linalg.svd(statics[0][:,:4],compute_uv=False)
    multi_singular = np.linalg.svd(static[:,:4],compute_uv=False)
    assert multi_singular[-1] > first_singular[-1]+1e-5
    pars = default_parameters()
    q = np.repeat(np.array(action_library()[0]["pose"])[None,:], 300, axis=0)
    dq = np.zeros_like(q)
    truth = {"mass": .9, "com": [.01,-.01,.07], "inertia_com": (np.eye(3)*.001).tolist()}
    known,y = reg.at(q[0],dq[0],dq[0])
    biases = np.array([.01,-.01,.02,-.02,.005,-.005,.015])
    torque = np.repeat((known+y@payload_to_pi(truth)+biases)[None,:],len(q),axis=0)
    torque += rng.normal(0,.01,torque.shape)
    log = {"q":q,"dq":dq,"torque":torque,"time":np.arange(len(q))*.01,
           "event_id":np.array([f'event-{i}' for i in range(len(q))]),"execution_id":"public-execution"}
    est = PayloadEvidenceEstimator(pars,regressor=reg)
    est.ingest(log)
    before = est.mean.copy(),est.covariance.copy()
    est.ingest(log)
    assert np.array_equal(before[0],est.mean) and np.array_equal(before[1],est.covariance)
    assert np.max(np.abs(est.covariance[4:10,4:10]-est._prior_cov[4:10,4:10])) < 1e-12
    split = PayloadEvidenceEstimator(pars,regressor=reg)
    for a,b in ((0,127),(127,len(q))):
        split.ingest({k:(v[a:b] if isinstance(v,np.ndarray) else v) for k,v in log.items()})
    assert np.max(np.abs(split.mean-est.mean)) < 1e-12
    assert np.max(np.abs(split.covariance-est.covariance)) < 1e-12
    # Identical numeric samples carrying fresh identities are legitimate evidence.
    newlog = {**log,"event_id":np.array([f'fresh-{i}' for i in range(len(q))]),"execution_id":"new-execution"}
    est.ingest(newlog)
    assert est.estimate()["unique_measurement_events"] == 600
    assert np.trace(est.covariance[:4,:4]) < np.trace(before[1][:4,:4])
    assert np.trace(est.covariance[10:,10:]) > 1e-7
    # Projection records its displacement and propagates uncertainty explicitly.
    malformed = payload_to_pi(truth)
    malformed[4:7] *= -1
    projected, meta = project_physical(malformed)
    assert meta["projected"] and np.linalg.eigvalsh(pseudo_inertia(projected))[0] >= -1e-12
    batch = _project_sample_matrix(np.stack([malformed,payload_to_pi(truth)]))
    assert np.max(np.abs(batch[0]-projected)) < 1e-12
    adaptive = adaptive_reading_check()
    assert adaptive["orders_change"], adaptive
    return {"passed": True, "scope": "public-model inverse dynamics plus synthetic linear-algebra invariants; no integrated experiment outcomes",
            "regressor_max_reconstruction_error_Nm":max(errors),
            "regressor_max_step_change":max(steps),
            "static_inertia_max_abs_column":float(np.max(np.abs(static[:,4:]))),
            "single_static_rank":rank_single,"multiple_static_rank":rank_multiple,
            "single_static_singular_values":first_singular.tolist(),
            "multiple_static_singular_values":multi_singular.tolist(),
            "identity_dedup":True,"numeric_duplicate_is_new_measurement":True,
            "file_chunk_invariance":True,"unobserved_inertia_covariance_preserved":True,
            "joint_bias_uncertainty_preserved":True,"physical_projection_displacement_recorded":True,
            "adaptive_reading_check":adaptive,"virtual_inverse_dynamics_calls":reg.inverse_calls+reg_half.inverse_calls,
            "integrated_simulated_seconds":0.,"wall_seconds":time.perf_counter()-start}


def adaptive_reading_check():
    """Same prior/design, different acquired linear observations -> next actions.

    This isolates decision dependence in a declared two-direction fixture. The
    real-run report independently checks whether actual source/test orders vary.
    """
    p = default_parameters()
    # Work decision1 probes mass and h_x in two distinct directions.
    yf = np.zeros((1,7,10)); yf[0,0,0]=.1; yf[0,1,1]=3.
    bank = {"Y_full":yf.tolist(),"Y_half":yf.tolist(),
            "known_full":np.zeros((1,7)).tolist(),"known_half":np.zeros((1,7)).tolist(),
            "torque_limits":[1]*7,"parameter_samples":128,"prospective_outcomes":16,
            "sampling_seed":194633,"impact_threshold":.05,"near_limit_ratio":.95,
            "actions":{}}
    for name,index,precision in (("mass_test",0,1e6),("com_test",1,1e8)):
        f=np.zeros((17,17)); f[index,index]=precision
        bank["actions"][name]={"information17":f.tolist(),"duration":8.,"kind":"static"}
    configured={"mass":.8,"com":[0,0,.07],"inertia_com":(np.eye(3)*.001).tolist()}
    covariance=np.diag(np.array([.07,.003,.001,.002,.0002,.0002,.0002,.0001,.0001,.0001]+[.02]*7)**2)
    prior_mean=np.r_[payload_to_pi(configured),np.zeros(7)]
    prior_covariance=covariance*2
    observation_covariance=covariance*2
    common_measurement_matrix=np.eye(17)
    precision=np.linalg.inv(prior_covariance)
    noise_precision=np.linalg.inv(observation_covariance)
    posterior_covariance=np.linalg.inv(precision+common_measurement_matrix.T@noise_precision@common_measurement_matrix)
    observed=[]
    for mass,com_x in ((1.29,0.),(.8,.0205),(.8,-.0205),(1.25,0.),(.8,.015)):
        target_mean=np.r_[payload_to_pi({**configured,"mass":mass,"com":[com_x,0,.07]}),np.zeros(7)]
        # Explicit common-prior/common-design update; only acquired y differs.
        measurement=2*target_mean-prior_mean
        est=PayloadEvidenceEstimator.__new__(PayloadEvidenceEstimator)
        est.parameters=p; est.geometry={}
        est.mean=posterior_covariance@(precision@prior_mean+common_measurement_matrix.T@noise_precision@measurement)
        est.covariance=posterior_covariance.copy()
        choice, detail=choose_action(est,list(bank["actions"]),bank,configured)
        observed.append({"measured_mass_direction":mass,"measured_com_direction":com_x,
                         "selected":choice,"reason":detail["reason"],"entropy":detail["prior_decision_entropy_bits"]})
    return {"orders_change":len({r["selected"] for r in observed})>1,
            "fixture":"explicit same linear-Gaussian prior/design/noise update; only acquired y differs",
            "rows":observed,"actual_experiment_order_dependence":"reported separately from executed runs"}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); result=checks(); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def run_checks(run_root):
    result=checks()
    output=Path(run_root)/"checks/estimator_checks.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


@lru_cache(maxsize=1)
def _actual_reading_context(run_root):
    root=Path(run_root)
    parameters=json.loads((root/"configs/estimator_parameters.json").read_text())
    bank=json.loads((root/"configs/design_bank.json").read_text())
    public=json.loads((root/"manifests/source_public.json").read_text())
    config=public[0]["configured_payload"]
    return parameters,bank,config


def _actual_reading_worker(args):
    run_root,case_id=args
    root=Path(run_root)
    parameters,bank,config=_actual_reading_context(run_root)
    path=root/"raw/source"/case_id/"block_00_L0"/"public.npz"
    metadata=json.loads(path.with_name("metadata.json").read_text())
    with np.load(path) as arrays:
        log={k:arrays[k] for k in ("q","dq","torque","time","event_id","unexpected_contact_flags")}
    log["execution_id"]=metadata["execution_id"]
    estimator=PayloadEvidenceEstimator(parameters,geometry=config)
    estimate=estimator.ingest(log)
    selected,rationale=choose_action(estimator,list(bank["actions"]),bank,config)
    return {"case_id":case_id,"public_record":str(path),"selected_action":selected,
            "reason":rationale["reason"],"decision_entropy_bits":rationale["prior_decision_entropy_bits"],
            "scores":rationale["scores"],"estimated_mass":estimate["payload_estimate"]["mass"],
            "estimated_com":estimate["payload_estimate"]["com"],
            "selection_cpu_seconds":rationale["selection_cpu_seconds"],
            "virtual_inverse_dynamics_calls":estimate["virtual_inverse_dynamics_calls"]}


def actual_source_reading_check(run_root, workers=12):
    """Replace only actual acquired L0 data under common prior/config/design.

    This is a selector-dependence check, not extra physical execution and not a
    claim that a commissioned method needed another test after its shared STOP.
    Only the source PUBLIC manifest and PUBLIC acquired records are read.
    """
    import hashlib
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    root=Path(run_root)
    cases=json.loads((root/"manifests/source_public.json").read_text())
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(_actual_reading_worker,[(str(root),c["case_id"]) for c in cases]))
    counts=dict(Counter(r["selected_action"] for r in rows))
    reasons=dict(Counter(r["reason"] for r in rows))
    result={"actual_acquired_source_histories":len(rows),"action_counts":counts,"reason_counts":reasons,
            "actions_change":len(counts)>1,"common_prior":True,"common_config":cases[0]["configured_payload"],
            "common_design":True,"only_input_changed":"actually acquired public source L0 q/dq/torque records",
            "interpretation":"Evidence dependence includes measured kinematics and resulting covariance. Entropy and variance-fallback counts are reported separately; this is not proof that mean-dependent entropy improves commissioning outcomes.",
            "counterfactual_next_action_after_L0":"Shared STOP is deliberately not applied in this selector-dependence check; deployed run outcomes are reported separately.",
            "parameter_sha256":hashlib.sha256((root/"configs/estimator_parameters.json").read_bytes()).hexdigest(),
            "bank_sha256":hashlib.sha256((root/"configs/design_bank.json").read_bytes()).hexdigest(),
            "virtual_inverse_dynamics_calls":sum(r["virtual_inverse_dynamics_calls"] for r in rows),
            "integrated_simulated_seconds":0.,"wall_seconds":time.perf_counter()-start,"workers":workers,"rows":rows}
    path=root/"checks/actual_source_reading_dependence.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    return {k:v for k,v in result.items() if k!="rows"}


if __name__=="__main__":
    main()
