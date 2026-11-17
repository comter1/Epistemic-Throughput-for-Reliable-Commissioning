"""Resumable stage interfaces. Commands dispatch physics to 2072 and 2379 only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .protocol import (REGISTERED,METHODS,HERE,EXP,REPO,write_json,file_hash,digest,
                       scientific_hashes,audit,verify_protected)
from .dispatch import dispatch,PYTHON

STAGES=('audit','qualify-source','fit','branch-checks','validate','seal','evaluate','report','package')

def config_hashes(root):
    root=Path(root)
    return {str(p.relative_to(root)):file_hash(p) for d in ('configs','manifests','private')
            for p in sorted((root/d).rglob('*.json')) if p.is_file()}

def asset_hashes():
    from payload_change_v1.physics import ASSET
    return {str(p):file_hash(p) for p in sorted(ASSET.rglob('*')) if p.is_file()}

def verify_seal(root):
    sealed=json.loads((Path(root)/'protocol_locked.json').read_text())
    if scientific_hashes()!=sealed['code_hashes']: raise RuntimeError('Scientific code differs from seal')
    if config_hashes(root)!=sealed['configuration_hashes']: raise RuntimeError('Configurations/manifests differ from seal')
    if asset_hashes()!=sealed['asset_hashes']: raise RuntimeError('Pinned robot assets differ from seal')
    verify_protected(root)

def jobs_for(root,stage):
    root=Path(root); split='validation' if stage=='validation' else 'test'
    cases=json.loads((root/f'private/{split}.json').read_text()); jobs=[]
    cfg=digest(config_hashes(root))
    if stage=='validation': multipliers=[1,2,4]; seeds=[7]; template_selected=0
    else:
        selection=json.loads((root/'configs/validation_selection.json').read_text())
        multipliers=[selection['multiplier']]; template_selected=selection['fixed_template']; seeds=[11,23,37]
    for case in cases:
        for seed in seeds:
            if stage=='reference':
                jobs.append(dict(stage=stage,key=f'{case["case_id"]}_{seed}',case=case,seed=seed,
                    config_hash=cfg,require_code_hash=True)); continue
            for multiplier in multipliers:
                for method in METHODS:
                    templates=range(3) if stage=='validation' and method=='fixed_prior_plan' else [template_selected if method=='fixed_prior_plan' else 0]
                    for template in templates:
                        key=f'{case["case_id"]}_{seed}_{method}_m{multiplier}_t{template}'
                        jobs.append(dict(stage=stage,key=key,case=case,seed=seed,method=method,multiplier=multiplier,
                            template=template,config_hash=cfg,require_code_hash=True))
    return jobs

def finish_validation(root):
    root=Path(root); results=[json.loads(p.read_text())['result'] for p in (root/'jobs/validation').glob('*.json') if not p.name.startswith('SOURCE_SMOKE')]
    assert len(results)==2304,(len(results),2304)
    df=pd.DataFrame([{k:v for k,v in r.items() if k not in ('steps','blocks','audits','estimate','updates','compute','motion_order','motion_speeds')} for r in results])
    df['physical_constraint_violation']=((df.identification_violations+df.verification_violations+df.audit_violations)>0)
    summary=df.groupby(['multiplier','method','fixed_template']).agg(n=('normal_correct','size'),normal_correct=('normal_correct','sum'),
        false_release=('false_release','sum'),violations=('physical_constraint_violation','sum'),capped_time=('capped_serial_seconds','mean'),
        identification_tracking_failures=('identification_tracking_failures','sum')).reset_index()
    summary.to_csv(root/'validation_all_methods.csv',index=False)
    fixed=summary[summary.method=='fixed_prior_plan']; fixed.to_csv(root/'validation_fixed_template_results.csv',index=False)
    coverage=pd.read_csv(root/'source_uncertainty_coverage.csv'); coverage=coverage[coverage.object_index>=6]
    base=json.loads((root/'configs/estimator_parameters.json').read_text())['posterior_covariance_inflation']; choices=[]
    for multiplier in (1,2,4):
        f=fixed[fixed.multiplier==multiplier].sort_values(['violations','false_release','normal_correct','capped_time','fixed_template'],ascending=[True,True,False,True,True]).iloc[0]
        selected=summary[(summary.multiplier==multiplier)&((summary.method!='fixed_prior_plan')|(summary.fixed_template==int(f.fixed_template)))]
        cov=dict(mass=float((coverage.mass_abs_z<=2.5758293035489004*np.sqrt(base*multiplier)).mean()),
            com=float((coverage.com_max_abs_z<=2.5758293035489004*np.sqrt(base*multiplier)).mean()),
            joint=float((coverage.parameter_mahalanobis_squared<=23.209251158954356*base*multiplier).mean()))
        okay=bool(min(cov.values())>=.99 and selected.false_release.sum()==0 and selected.violations.sum()==0)
        choices.append(dict(multiplier=multiplier,fixed_template=int(f.fixed_template),qualified=okay,source_coverage=cov,
            validation_false_releases=int(selected.false_release.sum()),validation_violations=int(selected.violations.sum())))
    chosen=next((x for x in choices if x['qualified']),choices[-1])
    write_json(root/'configs/validation_selection.json',chosen|dict(all_candidates=choices,
        rule='smallest multiplier with >=99% held-out-source mass/COM/joint coverage and zero observed validation physical constraint violations/false releases; otherwise explicitly unqualified multiplier 4',
        qualification_is_not_safety_certification=True,
        identification_tracking_note='Job accuracy tolerances are also tabulated on identification trajectories, separately from their contact/joint/torque safety constraints.'))
    return chosen

def forecast(root,stage):
    root=Path(root); totals=json.loads((root/'resources/totals.json').read_text())
    # Conservative physical bound includes all possible paid ID/verify and eight
    # half-speed audits. Prediction forecast is based on observed source histories,
    # with a fivefold allowance for later context/action branches, not an unbounded search.
    n=2304 if stage=='validate' else 3456
    jobs=list((root/'jobs/branches').glob('*.json'))
    rows=[json.loads(p.read_text())['result'] for p in jobs]
    early=sum(x['would_release_after_B0'] for x in rows)/max(len(rows),1)
    pred_per_run=5*(20+(1-early)*120)
    estimate=dict(stage=stage,all_case_physical_upper_seconds=n*240+(576*200 if stage=='evaluate' else 0),
        public_prediction_forecast_seconds=n*pred_per_run,
        cpu_forecast_core_hours=n*(10+(1-early)*70)/3600+5,
        observed_source_B0_release_fraction=early,source_histories=len(rows),current_totals={k:v['completed'] for k,v in totals.items()},
        hard_caps_enforced=True,forecast_is_not_measurement=True)
    write_json(root/'resources'/f'{stage}_forecast.json',estimate)
    if (estimate['all_case_physical_upper_seconds']+totals['plant']['charged']>2e6 or
        estimate['public_prediction_forecast_seconds']+totals['prediction']['charged']>1e6 or
        estimate['cpu_forecast_core_hours']+totals['cpu']['charged']/3600>160):
        raise RuntimeError('Forecast exceeds registered ceiling; package completed stages before requesting expansion')
    return estimate

def run_stage(stage,root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if stage=='audit':
        if (root/'checks/protected_hashes.json').exists(): result=verify_protected(root)
        else: result=audit(root)
        from .cases import generate_manifests
        from .workstations import build
        generate_manifests(root)
        if not (root/'configs/workstations.json').exists(): build(root)
        import mujoco,scipy,numpy
        write_json(root/'environment.json',dict(python=sys.version,mujoco=mujoco.__version__,scipy=scipy.__version__,numpy=numpy.__version__,
            inherited_controller=json.loads((__import__('payload_adaptive_tests_v2.protocol',fromlist=['PARENT']).PARENT/'configs/controller.json').read_text()),
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),allowed_ports=[2072,2379]))
        return result
    if stage=='qualify-source':
        from .protocol import BudgetExceeded
        cases=json.loads((root/'private/source.json').read_text())
        prof=sorted(cases,key=lambda c:c['case_id'])[:10]
        if len(list((root/'jobs/profile').glob('*.json')))<10:
            dispatch(root,[dict(stage='profile',key=c['case_id'],case=c) for c in prof],workers=(6,2),tag='profile')
        dispatch(root,[dict(stage='source',key=c['case_id'],case=c) for c in cases],workers=(136,44),tag='source')
        return dict(cases=288)
    if stage=='fit':
        from .fitting import initial_fit,finish_fit
        from .fixed_plans import build_banks
        initial_fit(root); cases=json.loads((root/'private/source.json').read_text())
        dispatch(root,[dict(stage='fit_case',key=c['case_id'],case=c,revision=2) for c in cases],workers=(136,44),tag='fit_case')
        result=finish_fit(root); build_banks(root); return result
    if stage=='branch-checks':
        from .branch_checks import selected_cases
        cases=selected_cases(root)
        write_json(root/'configs/source_branch_cases.json',[c['case_id'] for c in cases])
        dispatch(root,[dict(stage='branches',key=c['case_id'],case=c,revision=3) for c in cases],workers=(18,6),tag='branches')
        dispatch(root,[dict(stage='source_ablation',key=c['case_id'],case=c) for c in cases],workers=(18,6),tag='source_ablation')
        import shlex
        remote='cd '+shlex.quote(str(EXP))+' && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+PYTHON+' -m payload_adaptive_tests_v2.tests.engineering '+shlex.quote(str(root))
        with (root/'commands/engineering.log').open('w') as out:
            subprocess.run(['ssh','-o','BatchMode=yes','-p','2379','leiyo@ssh.cloud.sdu.dk',remote],stdout=out,stderr=subprocess.STDOUT,check=True)
        return dict(histories=24,engineering_tests=json.loads((root/'checks/engineering_tests.json').read_text())['passed'])
    if stage=='validate':
        assert json.loads((root/'checks/engineering_tests.json').read_text())['complete']
        forecast(root,stage); jobs=jobs_for(root,'validation')
        dispatch(root,jobs,workers=(136,44),tag='validation'); return finish_validation(root)
    if stage=='seal':
        assert len(list((root/'jobs/branches').glob('*.json')))==24
        selection=json.loads((root/'configs/validation_selection.json').read_text())
        verify_protected(root)
        write_json(root/'protocol_locked.json',dict(**REGISTERED,code_hashes=scientific_hashes(),configuration_hashes=config_hashes(root),
            asset_hashes=asset_hashes(),
            selection=selection,sealed_unix_time=time.time(),parent_commit='75fd606a4374a3101302a28c9231d293e1fa66e2',
            cache_contract='Shared exact deterministic public-model caches, keyed by all public inputs, code and MuJoCo/controller versions. Cache hits logged; actual CPU timing depends on warm/cold access and is not a standardized cold-latency benchmark. No acquired evidence reused as a new observation.'))
        return dict(sealed=True)
    if stage=='evaluate':
        verify_seal(root); forecast(root,stage)
        dispatch(root,jobs_for(root,'evaluation')+jobs_for(root,'reference'),workers=(136,44),tag='evaluation')
        verify_seal(root); return dict(main_runs=3456,reference_cases=576)
    if stage=='report':
        from .reporting import report
        verify_seal(root); return report(root)
    if stage=='package':
        from .packaging import package
        verify_seal(root); return package(root)
    raise ValueError(stage)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',required=True,choices=STAGES); ap.add_argument('--run-root',required=True)
    a=ap.parse_args(); root=Path(a.run_root); begin=time.time()
    result=run_stage(a.stage,root)
    marker=dict(stage=a.stage,status='complete',exit_code=0,finished_unix_time=time.time(),wall_seconds=time.time()-begin,
                code_hashes=scientific_hashes(),configuration_hashes=config_hashes(root),result=result)
    write_json(root/'stages'/(a.stage+'.json'),marker)
    print(json.dumps(dict(stage=a.stage,status='complete',result=result),default=str),flush=True)

if __name__=='__main__': main()
