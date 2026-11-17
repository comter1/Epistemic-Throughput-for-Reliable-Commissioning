"""Outcome-independent rigid weight layouts and strictly separate public views."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .protocol import digest, write_json, PARENT
from payload_change_v1.cases import make_payload

DATA_ROOT = Path('/work/Users/leiyo/cgx_data_v2')
SPLITS = {'source': (12, 8, [5]), 'validation': (4, 8, [7]), 'test': (8, 8, [11, 23, 37])}
MAIN_METHODS = ['original_task_once', 'fixed_four_pose', 'fixed_mixed_suite', 'adaptive_task_checks']
REFERENCES = ['keep_existing_settings', 'true_payload_settings_reference']


def generate_manifests(run_root):
    root = Path(run_root)
    if (root / 'manifests/test_public.json').exists():
        return
    # Family identity varies weight support topology, direction and known installation height.
    families = [(axis, layout, height) for axis in (0., np.pi / 2, np.pi / 4)
                for layout in ('beam_two_weights', 'rectangle_four_weights')
                for height in (.065, .075, .085, .095)]
    permutation = np.random.default_rng(918031).permutation(24)
    cursor = 0
    geometry_seen, task_seen = set(), set()
    for split, (nfamily, ncase, seeds) in SPLITS.items():
        private, public, family_rows = [], [], []
        for local_family in range(nfamily):
            family_index = int(permutation[cursor]); cursor += 1
            angle, layout, height = families[family_index]
            fid = digest({'family_structure': [angle, layout, height]})[:16]
            family_rows.append({'family_id': fid, 'axis_angle_rad': angle,
                                'support_topology': layout, 'mount_center_z_m': height})
            axis = np.array([np.cos(angle), np.sin(angle), 0.])
            transverse = np.array([-np.sin(angle), np.cos(angle), 0.])
            for index in range(ncase):
                group = 'G' + str(index % 4)
                seed = int(digest([2918032, split, family_index, index])[:15], 16)
                rng = np.random.default_rng(seed)
                size = rng.uniform([.12, .12, .04], [.16, .16, .065])
                center = np.array([0., 0., height])
                distance = rng.uniform(.024, .045)
                if layout == 'beam_two_weights':
                    points = np.array([center - distance * axis, center + distance * axis])
                    old_fraction, new_fraction = [.8, .2], [.2, .8]
                else:
                    points = np.array([center + sign * distance * axis + side * .009 * transverse
                                       for sign in (-1, 1) for side in (-1, 1)])
                    old_fraction, new_fraction = [.4, .4, .1, .1], [.1, .1, .4, .4]
                old_mass = float(rng.uniform(.3, 1.25))
                lower, upper = max(.6, .2 / old_mass), min(1.8, 1.5 / old_mass)
                # No outcome-based filtering and no requirement that stale settings fail.
                new_mass = old_mass * float(rng.uniform(lower, upper))
                old, old_geometry = make_payload(old_mass, size, center, points, old_fraction)
                true, true_geometry = make_payload(
                    old_mass if group in ('G0', 'G3') else new_mass, size, center, points,
                    new_fraction if group in ('G1', 'G3') else old_fraction)
                configured = copy.deepcopy(true if group in ('G0', 'G1') else old)
                geometry_key = digest([size.tolist(), center.tolist(), points.tolist()])
                task_seed = int(digest(['L0-task', seed])[:8], 16)
                verify_seed = int(digest(['public-verify', seed])[:8], 16)
                audit_seeds = [int(digest(['private-audit', seed, i])[:8], 16) for i in range(8)]
                if geometry_key in geometry_seen or any(s in task_seen for s in [task_seed, verify_seed, *audit_seeds]):
                    raise RuntimeError('split geometry/task stream collision')
                geometry_seen.add(geometry_key); task_seen.update([task_seed, verify_seed, *audit_seeds])
                cid = digest(['opaque_case', seed])[:20]
                row = {'case_id': cid, 'family_id': fid, 'split': split, 'group': group,
                       'noise_seeds': seeds, 'object_index': index, 'object_id': cid, 'old_payload': old, 'true_payload': true,
                       'configured_payload': configured, 'old_internal_geometry': old_geometry,
                       'true_internal_geometry': true_geometry, 'geometry_hash': geometry_key,
                       'task_seed': task_seed, 'public_verification_seed': verify_seed,
                       'private_audit_seeds': audit_seeds}
                for context in ('W0','W1','W2'):
                    task_seed = int(digest(['V2-public-work', context])[:8],16)
                    case_id = digest([cid,context])[:20]
                    private.append(row | {'object_id':cid, 'case_id':case_id, 'context':context,
                        'task_seed':task_seed})
                    public.append({'case_id':case_id, 'configured_payload':configured,
                        'mount_frame':'panda_hand', 'box_size':size.tolist(),
                        'box_center':center.tolist(), 'context':context, 'task_seed':task_seed})
        write_json(root / 'manifests' / f'{split}_public.json', public)
        write_json(root / 'private' / f'{split}.json', private)
        write_json(root / 'private' / f'{split}_families.json', family_rows)
    graph = json.loads((PARENT/'configs/healthy_graph.json').read_text())
    write_json(root/'configs/healthy_graph.json',graph)
    write_json(root / 'checks/split_integrity.json', {
        'passed': True, 'geometry_unique_count': len(geometry_seen), 'task_stream_unique_count': len(task_seen),
        'objects': {k:v[0]*v[1] for k,v in SPLITS.items()}, 'cases':{k:v[0]*v[1]*3 for k,v in SPLITS.items()}, 'family_count': 24,
        'public_numeric_feature_allowlist': ['configured_payload', 'box_size', 'box_center', 'mount_frame'],
        'case_id_usage': 'join/storage only, never a numeric estimator or policy feature'})


def policy_view(public_case):
    """Only physical public configuration reaches estimation/design."""
    cfg=public_case['configured_payload']
    return dict(configured_payload=copy.deepcopy(cfg),box_size=list(cfg['box_size']),
                box_center=list(cfg['box_center']),mount_frame='panda_hand',
                context=public_case['context'],task_seed=public_case['task_seed'])
