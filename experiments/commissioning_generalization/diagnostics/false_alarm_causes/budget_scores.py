"""Same initial evidence, B=1/B=4 public resources; frozen selector scores."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cgx.observation.actions import action_catalog, feasible_actions, initial_budget
from cgx.provenance import atomic_json, hash_json, read_json
from .core import inference_settings
from .interventions import Runner, METHODS, SEEDS
from .ledger import TimedStage


def initialize(root, run, device):
    global RUNNER
    inference_settings()
    RUNNER = Runner(root, run, device)


def score_case(case):
    r = RUNNER
    path = r.run/'budget_scores'/f"{case['case_id']}.parquet"
    if path.exists():
        return str(path)
    rows = []
    # Initial graph permutation and observations are common to all policies.
    initial, _ = r.initial(case, METHODS[0], SEEDS[0])
    q, _ = r.predict(initial, 'budget_initial')
    catalog = action_catalog(initial['graph'], r.config)
    record_hash = hash_json(initial['observations'])
    for method in METHODS:
        for seed in SEEDS:
            model = r.policy(method, seed)
            for cap in (1, 4):
                budget = initial_budget(initial['graph'], 1., cap)
                history = dict(initial, initial_budget=budget, remaining=dict(budget))
                legal = {a['action_key'] for a in feasible_actions(catalog, budget)}
                scores = r.score(model, history, catalog, q, 'budget_initial')
                chosen = max((i for i,a in enumerate(catalog) if a['action_key'] in legal), key=lambda i:scores[i])
                for index, (action, score) in enumerate(zip(catalog, scores)):
                    rows.append(dict(configuration_id=case['configuration_id'], case_id=case['case_id'],
                        method_key=method, policy_seed=seed, execution_cap=cap,
                        checkpoint_hash=model.checkpoint_hash, initial_observations_hash=record_hash,
                        action_key=action['action_key'], action_kind=action['kind'],
                        ports=json.dumps(action['ports']), sample_hz=action['sample_hz'],
                        calibration_pair=json.dumps(action['calibration_pair']),
                        legal=action['action_key'] in legal, score=float(score), selected=index==chosen,
                        resource_executions=budget['executions'], resource_bytes=budget['bytes'],
                        resource_calibrations=budget['calibrations'], resource_decisions=budget['decisions'],
                        effective_selector_decisions=min(2,budget['decisions']) if method=='gnn_depth2' else budget['decisions']))
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)


def run(root, run_dir, device='cuda:1', workers=4):
    run_dir=Path(run_dir);(run_dir/'budget_scores').mkdir(exist_ok=True)
    cases=read_json(run_dir/'manifests/discovery_cases.json')
    manifest=dict(case_ids=[c['case_id'] for c in cases], methods=list(METHODS), seeds=list(SEEDS),
                  caps=[1,4], selection='presealed 64 discovery cases; no outcome-dependent selection',
                  candidate_rule='all public catalog actions; legality separately recorded',
                  new_physics=False, score_calls_upper_bound=len(cases)*len(METHODS)*len(SEEDS)*2,
                  execution_host='ssh.cloud.sdu.dk:2982')
    manifest['manifest_hash']=hash_json(manifest)
    p=run_dir/'manifests/initial_budget_scores.json'
    if p.exists():
        assert read_json(p)==manifest
    else:
        atomic_json(p,manifest)
    with TimedStage(run_dir,'initial_budget_scores'):
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                initializer=initialize,initargs=(str(root),str(run_dir),device)) as pool:
            paths=[]
            for path in pool.map(score_case,cases):
                paths.append(path)
                print(json.dumps({'completed_cases':len(paths),'total_cases':len(cases)}),flush=True)
        data=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
        data.to_parquet(run_dir/'initial_budget_action_scores.parquet',index=False)
        keys=['configuration_id','case_id','method_key','policy_seed','action_key']
        pairs=data[data.execution_cap==1].merge(data[data.execution_cap==4],on=keys,suffixes=('_b1','_b4'),validate='one_to_one')
        summaries=[]
        for group,part in pairs.groupby(keys[:-1],sort=True):
            chosen1=part.loc[part.selected_b1,'action_key'].item()
            chosen4=part.loc[part.selected_b4,'action_key'].item()
            summaries.append(dict(zip(keys[:-1],group))|dict(first_action_b1=chosen1,first_action_b4=chosen4,
                first_action_changed=chosen1!=chosen4,
                legal_set_changed=bool((part.legal_b1!=part.legal_b4).any()),
                candidate_count=len(part), legal_count_b1=int(part.legal_b1.sum()),legal_count_b4=int(part.legal_b4.sum()),
                max_abs_score_change=float(np.abs(part.score_b4-part.score_b1).max()),
                mean_abs_score_change=float(np.abs(part.score_b4-part.score_b1).mean()),
                identical_initial_observations=bool((part.initial_observations_hash_b1==part.initial_observations_hash_b4).all())))
        pd.DataFrame(summaries).to_csv(run_dir/'initial_budget_score_pairs.csv',index=False)
        atomic_json(run_dir/'initial_budget_scores_status.json',dict(state='passed',cases=len(cases),paired_models=len(summaries),score_rows=len(data)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.data_root,a.run_dir,a.device,a.workers)
