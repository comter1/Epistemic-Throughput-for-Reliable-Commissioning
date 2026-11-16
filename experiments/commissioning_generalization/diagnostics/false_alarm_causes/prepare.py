"""Seal outcome-independent samples and task limits before new comparisons."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import socket
import tarfile
import numpy as np
from cgx.config import EXPERIMENT, load_config
from cgx.provenance import atomic_json, file_sha256, hash_json
from cgx.graphs.manifests import read_jsonl
from cgx.seeds import seed_for
from cgx.env.system import generate_task


def prepare(root, run):
    root, run = Path(root), Path(run)
    if (run/'INPUT_MANIFEST.json').exists():
        raise RuntimeError('This investigation is already sealed; do not overwrite it')
    config = load_config()
    cases = list(read_jsonl(root/'private/cases/test_connections.jsonl'))
    graphs = {r['configuration_id']:r['graph'] for r in read_jsonl(root/'manifests/public_graphs.jsonl')}
    by_config = defaultdict(lambda:defaultdict(list))
    for case in cases:by_config[case['configuration_id']][case['fault_family']].append(case)
    families = ['cache_stale','queue_hold','adapter_scale']
    replay, discovery, fresh = [], [], []
    for ci,cid in enumerate(sorted(by_config)):
        groups=by_config[cid]
        for family in groups:groups[family].sort(key=lambda c:c['case_id'])
        replay.extend(groups['healthy'])
        for family in families:replay.extend(groups[family][:2])
        discovery.extend(groups['healthy'][:2])
        discovery.extend(groups[families[(ci+j)%3]][0] for j in range(2))
        for j in range(8):
            family='healthy' if j<4 else families[(ci+j-4)%3]
            graph=graphs[cid]
            ident=hash_json(['CGX-FA-CAUSE-V1',671920,cid,family,j])
            streams={name:seed_for(ident,name) for name in cases[0]['streams']}
            rng=np.random.default_rng(seed_for(ident,'fault'))
            ntype={'cache_stale':'cache','queue_hold':'queue','adapter_scale':'adapter'}.get(family)
            legal=[n['id'] for n in graph['nodes'] if n['type']==ntype]
            location=legal[int(rng.integers(len(legal)))] if legal else None
            if family=='adapter_scale':
                values=[s*m for m in config['faults']['adapter_scale_magnitudes'] for s in config['faults']['adapter_scale_signs']]
            elif family=='cache_stale':values=config['faults']['cache_stale_delay_ticks']
            elif family=='queue_hold':values=config['faults']['queue_hold_delay_ticks']
            else:values=[None]
            severity=values[int(rng.integers(len(values)))]
            clock=np.random.default_rng(streams['clock'])
            offsets=dict(zip(graph['clock_domains'],map(float,clock.uniform(*config['sensing']['observer_offset_uniform_ms'],len(graph['clock_domains'])))))
            fresh.append(dict(case_id=ident,configuration_id=cid,split='test_connections',
                fault_family=family,fault_node=location,severity=severity,observer_offsets_ms=offsets,
                streams=streams,example_only=False,investigation_only=True))
    assert len(replay)==496 and len(discovery)==64 and len(fresh)==128
    assert len({c['case_id'] for c in replay})==496
    assert not {c['case_id'] for c in fresh}&{c['case_id'] for c in cases}
    for name,rows in [('replay_cases',replay),('discovery_cases',discovery),('fresh_cases',fresh)]:
        atomic_json(run/'manifests'/f'{name}.json',rows)
    atomic_json(run/'manifests/fresh_tasks.json',{c['case_id']:generate_task(c['streams']['task']) for c in fresh})
    methods=['gnn_depth2','gnn_adaptive','transformer_adaptive','gnn_fixed_plan']
    jobs=[dict(case_id=c['case_id'],configuration_id=c['configuration_id'],method=m,policy_seed=s,
               execution_cap=4,artifact_key=hash_json([m,s,4,1.0,c['case_id']]))
          for c in replay for m in methods for s in [11,23,37]]
    atomic_json(run/'manifests/replay_jobs.json',jobs)
    selection=json.loads((root/'models/selection.json').read_text())
    frozen=json.loads((root/'models/frozen.json').read_text())
    protected={}
    for name,expected in selection['files'].items():
        actual=file_sha256(root/name)
        if actual!=expected:raise ValueError(f'Changed selected model artifact: {name}')
        protected[name]=actual
    for name,expected in json.loads((root/'results/hashes.json').read_text()).items():
        if file_sha256(root/name)!=expected:raise ValueError(f'Changed result: {name}')
        protected[name]=expected
    for name in ['models/selection.json','models/frozen.json','models/thresholds.json','protocol_lock.json',
                 'software/source.tar.gz','software/source_files.json','software/protocol.json',
                 'results/hashes.json','private/cases/test_connections.jsonl','manifests/public_graphs.jsonl',
                 'manifests/split_manifest.jsonl','manifests/tasks.jsonl']:
        protected[name]=file_sha256(root/name)
    source=json.loads((root/'software/source_files.json').read_text())
    differences=[]
    for name,old in source.items():
        current=EXPERIMENT/name
        new=file_sha256(current) if current.exists() else None
        if old!=new:differences.append(dict(path=name,sealed_sha256=old,current_sha256=new))
    scientific=[r for r in differences if r['path'].startswith('src/cgx/') and not r['path'].startswith('src/cgx/reporting/')]
    if scientific:raise ValueError(f'Scientific code differs from sealed run: {scientific}')
    assert file_sha256(EXPERIMENT/'src/cgx/models/features.py')==frozen['preprocessing_hash']
    atomic_json(run/'provenance/source_comparison.json',dict(differences=differences,
        scientific_code_matches_sealed_run=True,preprocessing_matches_frozen=True))
    with tarfile.open(run/'provenance/workspace_before_analysis.tar.gz','w:gz') as archive:
        for path in sorted(EXPERIMENT.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts and 'diagnostics' not in path.relative_to(EXPERIMENT).parts:
                archive.add(path,arcname=str(path.relative_to(EXPERIMENT)))
    protocol={
        'task':'CGX-FA-CAUSE-V1','sealed_at':datetime.now(timezone.utc).isoformat(),
        'sample_selection':'lexicographic case hash within configuration/family, independent of outcomes',
        'replay':{'cases':496,'healthy':400,'faulty':96,'methods':methods,'seeds':[11,23,37],
                  'trajectories':5952,'max_prefixes':7,'readers':['final','independent']},
        'discovery':{'cases':64,'healthy':32,'faulty':32,'methods':['gnn_depth2','gnn_adaptive'],
                     'seeds':[11,23,37],'arms':['A','B','C','D'],'execution_cap':4,
                     'calibration_cap':2,'decision_cap':6},
        'arms':{'A':'current evidence and current q, freely chosen calibrations',
                'B':'selector retains initial observations and q; actual remaining/resources/action_keys',
                'C':'A plus up to two preregistered public-clock calibrations at start',
                'D':'B plus exactly the same preregistered calibrations as C'},
        'calibration_rule':'initial highest-score feasible ordinary action; public cross-domain coverage then fixed hash; public connection fallback; no labels or offsets',
        'calibration_tie_seed':671921,'bootstrap_seed':671922,'bootstrap_repetitions':2000,
        'fresh':{'max_cases':128,'healthy_per_config':4,'faulty_per_config':4,'generation_seed':671920,
                 'same_structures_new_streams':True,'maximum_confirmed_causes':2,
                 'selection_rule':'prefer isolated code or same-transcript evidence; then magnitude of paired healthy-FA reduction with fault-miss change disclosed; never p-value ranking',
                 'resource_fallback':'confirm one cause first; do not silently change locked discovery/replay denominators; only balanced configuration-complete fresh subsets allowed before confirmation launch'},
        'branches':{'max_prefixes':32,'max_actions':4,'noise_replicates':3,'seed':671923,
                    'selection':'hash-selected, 16 per method, balanced healthy/faulty across configurations; no outcome selection',
                    'full_Q_comparison':'same frozen continuation until the modeled remaining-decision horizon, paired across actions'},
        'limits':{'new_robot_execution_attempts':18000,'combined_model_forward_and_feature_build_attempts':120000,
                  'gpu_hours_soft':24,'cpu_core_hours_soft':200,'wide_call_watchdog_seconds':240},
        'cache_boundary':'Only after an action is chosen may an exactly matching case/streams/action/execution_index/calibration_index observation be reused; logical resource cost is still charged. Shared original initial records and immutable acquired-prefix features may be reused. Cache hits are reported separately.',
        'statistics':'average seeds per physical case; equal configurations; paired configuration-then-case bootstrap; retain failures and missingness',
        'compute':{'primary_node_port':2404,'replay_gpus':[0,1],'intervention_gpus':[2,3],
                   'input_training_node_port':2379,'input_training_gpus':[0],
                   'avoid_busy_node_port':2982,'local_role':'coordination, code editing, final packaging'},
        'code_changes':'new diagnostics only; old protocol, models, results and thresholds immutable',
    }
    # JSON is a strict subset of YAML, readable by either parser without a new dependency.
    (run/'PROTOCOL_FA_CAUSE_V1.yaml').write_text(json.dumps(protocol,indent=2)+'\n')
    manifest={'task':protocol['task'],'data_root':str(root),'run_dir':str(run),'host':socket.gethostname(),
        'sealed_at':protocol['sealed_at'],'original_hashes':protected,
        'source_comparison':'provenance/source_comparison.json',
        'sample_manifest_hashes':{str(p.relative_to(run)):file_sha256(p) for p in sorted((run/'manifests').glob('*.json'))},
        'protocol_sha256':file_sha256(run/'PROTOCOL_FA_CAUSE_V1.yaml'),
        'method_display_names':json.loads((root/'presentation_method_names/method_display_names.json').read_text())}
    atomic_json(run/'INPUT_MANIFEST.json',manifest)
    (run/'WORKLOAD_AND_STATUS.md').write_text(
        '# CGX-FA-CAUSE-V1 workload\n\nSamples and intervention rules were sealed before new comparisons.\n\n'
        '- Original replay: 5,952 B=4 trajectories, at most 7 prefixes, two frozen readers; immutable-prefix caching.\n'
        '- Discovery: 64 cases × 2 selectors × 3 seeds × 4 arms = 1,536 investigations.\n'
        '- Confirmation: at most 128 fresh cases; at most two isolated causes, bounded by remaining ledger.\n'
        '- Value checks: at most 32 prefixes × 4 actions × 3 measurement replicates, with matched continuation.\n'
        '- Hard limits: 18,000 new physical execution attempts; 120,000 combined forward/feature-build attempts.\n'
        '- Original observations/models/results are read-only. All failures and retries consume the ledger.\n\n'
        'Engineering throughput measurements will be appended before large dispatch. Status is recorded in stage JSON/log files.\n')
    print(json.dumps({'run_dir':str(run),'protected_files':len(protected),'replay_cases':len(replay),
                      'discovery_cases':len(discovery),'fresh_cases':len(fresh),'scientific_source_matches':True},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();prepare(args.data_root,args.run_dir)
