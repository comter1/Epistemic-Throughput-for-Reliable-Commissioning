"""Outcome-independent rigid weight layouts and strictly separate public views."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .storage import digest, write_json

DATA_ROOT = Path('/work/Users/leiyo/cgx_data_v2')
SPLITS = {'source': (12, 16, [5]), 'validation': (4, 16, [7]), 'test': (8, 24, [11, 23, 37])}
MAIN_METHODS = ['original_task_once', 'fixed_four_pose', 'fixed_mixed_suite', 'adaptive_task_checks']
REFERENCES = ['keep_existing_settings', 'true_payload_settings_reference']


def make_payload(mass, size, center, points, fractions, radius=.004):
    """12% uniform box + 88% rigid internal spherical weights; exact moments."""
    size, center, points = map(np.asarray, (size, center, points))
    fractions = np.asarray(fractions, dtype=float)
    fractions /= fractions.sum()
    weights = np.r_[.12, .88 * fractions] * mass
    locations = np.vstack([center, points])
    com = (weights[:, None] * locations).sum(axis=0) / mass
    inertia = np.zeros((3, 3))
    for i, (weight, location) in enumerate(zip(weights, locations)):
        local = (weight / 12 * np.diag(np.sum(size ** 2) - size ** 2)
                 if i == 0 else np.eye(3) * .4 * weight * radius ** 2)
        displacement = location - com
        inertia += local + weight * (np.dot(displacement, displacement) * np.eye(3)
                                     - np.outer(displacement, displacement))
    if not np.all(np.abs(points - center) + radius <= size / 2 + 1e-12):
        raise ValueError('internal weights outside shell')
    return {'mass': float(mass), 'com': com.tolist(), 'inertia_com': inertia.tolist(),
            'box_size': size.tolist(), 'box_center': center.tolist(), 'frame': 'panda_hand',
            'units': 'kg,m,kg*m^2'}, {
                'shell_mass': float(weights[0]), 'weight_masses': weights[1:].tolist(),
                'weight_centers': points.tolist(), 'weight_radius': radius}


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
                seed = int(digest([918032, split, family_index, index])[:15], 16)
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
                       'noise_seeds': seeds, 'old_payload': old, 'true_payload': true,
                       'configured_payload': configured, 'old_internal_geometry': old_geometry,
                       'true_internal_geometry': true_geometry, 'geometry_hash': geometry_key,
                       'task_seed': task_seed, 'public_verification_seed': verify_seed,
                       'private_audit_seeds': audit_seeds}
                private.append(row)
                public.append({'case_id': cid, 'configured_payload': configured,
                               'mount_frame': 'panda_hand', 'box_size': size.tolist(),
                               'box_center': center.tolist(), 'task_seed': task_seed,
                               'public_verification_seed': verify_seed})
        write_json(root / 'manifests' / f'{split}_public.json', public)
        write_json(root / 'private' / f'{split}.json', private)
        write_json(root / 'private' / f'{split}_families.json', family_rows)
    candidates = [json.loads(line) for line in (DATA_ROOT / 'manifests/public_graphs.jsonl').read_text().splitlines()
                  if json.loads(line)['configuration_id'].startswith('train-')]
    chosen = min(candidates, key=lambda row: digest(row['graph']))
    graph = copy.deepcopy(chosen['graph'])
    # Same executed graph and nominal transport, command representation is now torque.
    for node in graph['nodes']:
        for port in node['ports']:
            if port.get('unit') == 'command_rad':
                port['unit'] = 'command_Nm'
    write_json(root / 'configs/healthy_graph.json', graph)
    write_json(root / 'configs/graph_selection.json', {
        'rule': 'minimum canonical SHA256 among source public graphs; no outcomes inspected',
        'source_configuration_id': chosen['configuration_id'],
        'original_graph_sha256': digest(chosen['graph']), 'torque_graph_sha256': digest(graph),
        'only_change': 'command_rad port labels become command_Nm; topology/operators unchanged'})
    write_json(root / 'checks/split_integrity.json', {
        'passed': True, 'geometry_unique_count': len(geometry_seen), 'task_stream_unique_count': len(task_seen),
        'cases': {k: v[0] * v[1] for k, v in SPLITS.items()}, 'family_count': 24,
        'public_numeric_feature_allowlist': ['configured_payload', 'box_size', 'box_center', 'mount_frame'],
        'case_id_usage': 'join/storage only, never a numeric estimator or policy feature'})


def policy_view(public_case):
    """Only physical public configuration reaches estimation/design."""
    return {key: copy.deepcopy(public_case[key]) for key in
            ('configured_payload', 'box_size', 'box_center', 'mount_frame')}
