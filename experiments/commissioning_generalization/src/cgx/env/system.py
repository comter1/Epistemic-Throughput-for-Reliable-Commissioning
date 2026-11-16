"""Headless Panda execution. Private physics never crosses the observation boundary."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import copy
import hashlib
import os
import time
import numpy as np
import networkx as nx
from .modules import Packet, Module, merge_packets
from ..observation.actions import resources, trace_ports
from ..observation.clocks import stable_seed, calibration_burst
from ..observation.wire import encode_record, encode_burst
from ..observation.protocol import require_supported

HOME = np.array([0., -.7853981634, 0., -2.35619449, 0., 1.570796327, .7853981634])
DEADLINES = np.array([3.2, 6.4, 9.6])

@lru_cache(maxsize=2)
def _load_model(xml):
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml)
    model.opt.timestep = .002
    return model

@dataclass
class ExecutionResult:
    observation: dict
    wire: bytes
    metrics: dict
    debug: dict = field(default_factory=dict)

def generate_task(seed):
    """Source-independent joint-space catalog distribution with reachable FK targets."""
    rng = np.random.default_rng(seed)
    # Small but nontrivial reachable neighborhood, common across all configurations.
    amplitudes = np.array([.14, .10, .14, .10, .12, .10, .14])
    waypoints = HOME[None, :] + rng.uniform(-1, 1, (3, 7)) * amplitudes
    return {'home': HOME.tolist(), 'waypoints': waypoints.tolist(), 'deadlines_s': DEADLINES.tolist()}

def reference_at(t, task):
    waypoints = np.asarray(task['waypoints'], dtype=float)
    home = np.asarray(task.get('home', HOME), dtype=float)
    deadlines = np.asarray(task.get('deadlines_s', DEADLINES), dtype=float)
    previous, start = home, .5
    for target, deadline in zip(waypoints, deadlines):
        if t <= deadline:
            s = np.clip((t - start) / .1, 0., 1.)
            s = s * s * (3 - 2 * s)
            return previous + s * (target - previous)
        previous, start = target, deadline
    return waypoints[-1]

class System:
    """Evaluator/collector object; pass only result.observation to a policy."""
    def __init__(self, graph, private_case, task, config=None, asset_path=None, implementation='A'):
        import mujoco
        self.mujoco = mujoco
        self.graph = copy.deepcopy(graph)
        self.case = copy.deepcopy(private_case)
        self.task = copy.deepcopy(task)
        self.config = require_supported(config)
        if implementation not in ('A', 'B'):
            raise ValueError('unknown implementation')
        self.implementation = implementation
        if asset_path is None:
            asset_path = Path(os.environ.get('CGX_DATA_ROOT', '/home/leiyo/work/cgx_data')) / 'assets/mujoco_menagerie/franka_emika_panda'
        asset_path = Path(asset_path)
        xml = asset_path if asset_path.suffix == '.xml' else asset_path / 'scene.xml'
        if not xml.exists():
            raise FileNotFoundError(f'Pinned Panda asset absent: {xml}')
        self.model = _load_model(str(xml.resolve()))
        self.data = mujoco.MjData(self.model)
        self.clean = mujoco.MjData(self.model)
        self.nodes = {n['id']: n for n in self.graph['nodes']}
        self.modules = {nid: Module(node, implementation) for nid, node in self.nodes.items()}
        dag = nx.DiGraph()
        dag.add_nodes_from(self.nodes)
        dag.add_edges_from((e['source'], e['target']) for e in self.graph['edges'])
        self.order = list(nx.topological_sort(dag))
        self.incoming = {nid: [] for nid in self.nodes}
        for edge in self.graph['edges']:
            self.incoming[edge['target']].append(edge)
        self.ports = trace_ports(graph)
        self.port_indices = {port: idx for idx, port in enumerate(self.ports + ['external:out'])}
        self.offsets = dict(self.case.get('observer_offsets_ms', {}))
        self.external_domain = graph['clock_domains'][0]
        if set(self.offsets) != set(graph['clock_domains']):
            raise ValueError('private clock offsets must cover all public domains')
        self.streams = self.case.get('streams', {})
        self.lower = self.model.jnt_range[:7, 0].copy()
        self.upper = self.model.jnt_range[:7, 1].copy()
        self.ee_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        if self.ee_body < 0:
            raise ValueError('Panda hand body absent')
        self.target_xyz = np.array([self._fk(q) for q in self.task['waypoints']])
        family = self.case.get('fault_family')
        fault_node = self.case.get('fault_node')
        family_types = {'cache_stale': 'cache', 'queue_hold': 'queue', 'adapter_scale': 'adapter'}
        if family == 'healthy':
            if fault_node is not None or self.case.get('severity') is not None:
                raise ValueError('healthy case contains injected configuration')
        elif family in family_types:
            if fault_node not in self.nodes or self.nodes[fault_node]['type'] != family_types[family]:
                raise ValueError('fault is not applicable at this component')
            severity = self.case.get('severity')
            valid = (severity in (2, 5, 10)) if family != 'adapter_scale' else (severity in (-.15, -.1, -.05, .05, .1, .15))
            if not valid:
                raise ValueError('fault severity is outside the frozen catalog')
        else:
            raise ValueError('unknown fault family')
        if fault_node is not None:
            module = self.modules[fault_node]
            family = self.case['fault_family']
            if family in ('cache_stale', 'queue_hold'):
                module.delay_ticks = int(self.case['severity'])
            elif family == 'adapter_scale':
                module.extra_gain = 1. + float(self.case['severity'])
            else:
                raise ValueError(f'unknown family {family}')
        self.repair_log = []

    def _fk(self, q):
        self.mujoco.mj_resetData(self.model, self.clean)
        self.clean.qpos[:7] = q
        self.clean.qpos[7:] = .04
        self.mujoco.mj_forward(self.model, self.clean)
        return self.clean.xpos[self.ee_body].copy()

    def set_task(self, task, noise_seed):
        """Fresh public verification/audit task; addressed repairs remain in place."""
        self.task = copy.deepcopy(task)
        self.target_xyz = np.array([self._fk(q) for q in self.task['waypoints']])
        self.streams = {**self.streams, 'noise': int(noise_seed)}

    def repair(self, node_id):
        """Addressed ordinary configuration reset, with no fault-location lookup."""
        if node_id not in self.modules:
            raise ValueError('unknown repair node')
        module = self.modules[node_id]
        if module.node['type'] not in ('cache', 'queue', 'adapter'):
            raise ValueError('node has no registered repair interface')
        start = time.perf_counter()
        before = (module.delay_ticks, module.extra_gain)
        module.repair()
        result = {'node': node_id, 'configuration_changed': before != (0, 1.),
                  'wall_seconds': time.perf_counter() - start}
        self.repair_log.append(result)
        return result

    def run(self, ports=(), sample_hz=10, execution_index=0, debug=False):
        if sample_hz not in (10, 50, 100):
            raise ValueError('sample rate not in frozen catalog')
        selected = set(ports)
        if not selected.issubset(self.port_indices) or 'external:out' in selected:
            raise ValueError('unknown internal probe port')
        start = time.perf_counter()
        mujoco, model, data = self.mujoco, self.model, self.data
        mujoco.mj_resetData(model, data)
        home = np.asarray(self.task.get('home', HOME))
        data.qpos[:7] = home
        data.qpos[7:] = .04
        data.ctrl[:7] = home
        data.ctrl[7:] = 255.
        mujoco.mj_forward(model, data)
        for module in self.modules.values():
            module.reset()
        noise_seed = int(self.streams.get('noise', 0))
        rng = np.random.default_rng(stable_seed(noise_seed, 'plant', execution_index))
        qnoise = rng.normal(0, 1e-4, (1000, 7))
        vnoise = rng.normal(0, 1e-3, (1000, 7))
        external_noise = np.random.default_rng(stable_seed(noise_seed, 'external', execution_index)).normal(0, 1e-4, (1000, 3))
        token_rng = np.random.default_rng(stable_seed(noise_seed, 'tokens', execution_index))
        sensor_tokens = token_rng.bytes(16 * 1000)
        command_tokens = token_rng.bytes(16 * 1000)
        selected_all = selected | {'external:out'}
        next_sample, timestamp_errors, sample_counts = {}, {}, {}
        port_specs = {f"{n['id']}:{p['name']}": (n, p) for n in self.graph['nodes'] for p in n['ports']}
        for port in selected_all:
            domain = self.external_domain if port == 'external:out' else port_specs[port][0]['clock_domain']
            rate = 10 if port == 'external:out' else sample_hz
            arm = np.random.default_rng(stable_seed(noise_seed, 'arm', execution_index, domain))
            next_sample[port] = float(arm.uniform(0, .1) + arm.uniform(0, 1 / rate))
            timestamp_errors[port] = np.random.default_rng(stable_seed(noise_seed, 'timestamp', execution_index, port)).normal(0, .001, 1000)
            sample_counts[port] = 0
        records, wire_parts = [], []
        debug_data = {'controller_inputs': [], 'commands': [], 'actuator_commands': [], 'port_times': [], 'physical_q': []} if debug else {}
        recorder_seconds = 0.
        errors = np.zeros((1000, 3))
        joint_violation = command_violation = contact_count = 0
        external_pass = True
        def record(port, packet, tick):
            nonlocal recorder_seconds
            if port not in selected_all:
                return
            t = tick * .01 + (.01 if port == 'external:out' else 0.)
            if t + 1e-10 < next_sample[port]:
                return
            t0 = time.perf_counter()
            rate = 10 if port == 'external:out' else sample_hz
            next_sample[port] += 1 / rate
            count = sample_counts[port]
            sample_counts[port] += 1
            if port == 'external:out':
                domain, channels, event = self.external_domain, tuple(range(7)), 'sample'
            else:
                node, spec = port_specs[port]
                domain, channels = node['clock_domain'], spec['channels']
                event = 'receive' if spec['direction'] == 'in' else 'send'
            rec = dict(port=port, local_time_s=float(t + self.offsets[domain] / 1000 + timestamp_errors[port][tick]),
                       message_token=packet.token.hex(), event_type=event, values=np.asarray(packet.payload(), dtype=np.float32).tolist(), missing=bool(packet.missing))
            records.append(rec)
            wire_parts.append(encode_record(rec, self.port_indices[port], channels))
            if debug:
                debug_data['port_times'].append({'port': port, 'true_time': t, 'error': float(timestamp_errors[port][tick]),
                                                'local_time': rec['local_time_s']})
            recorder_seconds += time.perf_counter() - t0
        initial_values = np.zeros(14)
        initial_values[:7] = home
        initial_command = Packet(initial_values, b'\0' * 16, tuple(range(7)), False, missing=True)
        for tick in range(1000):
            t = tick * .01
            outputs = {}
            ref = reference_at(t, self.task)
            for nid in self.order:
                node, module = self.nodes[nid], self.modules[nid]
                inputs = {}
                for edge in self.incoming[nid]:
                    packet = outputs[(edge['source'], edge['source_port'])]
                    inputs[edge['target_port']] = packet
                    record(f"{nid}:{edge['target_port']}", packet, tick)
                kind = node['type']
                if kind == 'sensor':
                    values = np.concatenate((data.qpos[:7] + qnoise[tick], data.qvel[:7] + vnoise[tick]))
                    out = Packet(values, sensor_tokens[16*tick:16*(tick+1)], tuple(range(7)), True)
                elif kind == 'merger':
                    out = merge_packets(list(inputs.values()))
                else:
                    if len(inputs) != 1:
                        raise ValueError(f'{kind} requires one input, got {len(inputs)}')
                    packet = next(iter(inputs.values()))
                    if kind == 'cache':
                        out = module.cache(packet, tick)
                    elif kind == 'adapter':
                        out = module.adapter(packet)
                    elif kind == 'queue':
                        out = module.queue(packet, tick, initial_command.subset(packet.channels))
                    elif kind == 'splitter':
                        out = packet
                    elif kind == 'controller':
                        measured = packet.canonical()
                        command = measured[:7] + .20 * (ref - measured[:7]) - .02 * measured[7:14]
                        command = np.clip(command, self.lower + .001, self.upper - .001)
                        values = np.zeros(14)
                        values[:7] = command
                        out = Packet(values, command_tokens[16*tick:16*(tick+1)], tuple(range(7)), False)
                        if debug:
                            debug_data['controller_inputs'].append(measured.tolist())
                            debug_data['commands'].append(command.tolist())
                    elif kind == 'actuator':
                        command = packet.canonical()[:7]
                        command_violation += int(np.any(command < model.actuator_ctrlrange[:7, 0] - 1e-10) or np.any(command > model.actuator_ctrlrange[:7, 1] + 1e-10))
                        data.ctrl[:7] = np.clip(command, model.actuator_ctrlrange[:7, 0], model.actuator_ctrlrange[:7, 1])
                        out = packet
                        if debug:
                            debug_data['actuator_commands'].append(data.ctrl[:7].tolist())
                    else:
                        raise ValueError(f'unknown node type {kind}')
                for spec in node['ports']:
                    if spec['direction'] == 'out':
                        packet = out.subset(spec['channels'])
                        outputs[(nid, spec['name'])] = packet
                        record(f"{nid}:{spec['name']}", packet, tick)
            for _ in range(5):
                mujoco.mj_step(model, data)
                joint_violation += int(np.any(data.qpos[:7] < self.lower - .001) or np.any(data.qpos[:7] > self.upper + .001))
                contact_count += int(data.ncon)
            # mj_step integrates qpos after its position stage; refresh kinematics
            # so the external event timestamp refers to the actual measured pose.
            mujoco.mj_kinematics(model, data)
            xyz = data.xpos[self.ee_body].copy()
            errors[tick] = np.linalg.norm(self.target_xyz - xyz, axis=1)
            ref_xyz = self._fk(ref)
            ext_values = np.zeros(14)
            ext_values[:3] = xyz + external_noise[tick]
            ext_values[3:6] = ref_xyz
            ext_values[6] = float(np.linalg.norm(ext_values[:3] - ref_xyz) <= .02)
            ext_values[7] = float(joint_violation == 0)
            ext_values[8] = float(contact_count == 0)
            ext_values[9] = float(command_violation == 0)
            # External payload is deployable measured position/reference and requirement flags only.
            ext = Packet(ext_values, sensor_tokens[16*tick:16*(tick+1)], tuple(range(7)), True)
            record('external:out', ext, tick)
            if debug:
                debug_data['physical_q'].append(data.qpos[:7].tolist())
        hold_pass, hold_max = [], []
        deadlines = self.task.get('deadlines_s', DEADLINES)
        for i, deadline in enumerate(deadlines):
            lo, hi = int(round((deadline - .5) * 100)), int(round(deadline * 100))
            maximum = float(errors[lo:hi, i].max())
            hold_max.append(maximum)
            hold_pass.append(maximum <= .02)
        passed = all(hold_pass) and joint_violation == 0 and command_violation == 0 and contact_count == 0
        wire = b''.join(wire_parts)
        wall = time.perf_counter() - start
        observation = dict(action_key='initial' if not ports else 'run', records=records, calibration_bursts=[],
                           visible_status='ok', actual_cost=resources(1, len(wire), 0, 1))
        metrics = dict(task_pass=passed, waypoint_hold_pass=hold_pass, waypoint_hold_max_error_m=hold_max,
                       joint_limit_violations=joint_violation, command_limit_violations=command_violation,
                       prohibited_contacts=contact_count, wall_seconds=wall, recorder_seconds=recorder_seconds,
                       simulation_seconds=wall-recorder_seconds, simulated_seconds=10., record_count=len(records))
        return ExecutionResult(observation, wire, metrics, debug_data)

    def calibrate(self, pair, burst_index=0):
        if len(pair) != 2 or pair[0] == pair[1] or any(d not in self.offsets for d in pair):
            raise ValueError('invalid calibration pair')
        start = time.perf_counter()
        burst, duration = calibration_burst(pair, self.offsets, self.streams.get('noise', 0), burst_index)
        wire = encode_burst(burst, {d: i for i, d in enumerate(self.graph['clock_domains'])})
        observation = dict(action_key='calibration', records=[], calibration_bursts=[burst], visible_status='ok',
                           actual_cost=resources(0, len(wire), 1, 1))
        return ExecutionResult(observation, wire, {'wall_seconds': time.perf_counter()-start, 'simulated_seconds': duration})

    def acquire(self, action, execution_index=1, burst_index=0):
        if action['kind'] == 'stop':
            return ExecutionResult(dict(action_key=action['action_key'], records=[], calibration_bursts=[],
                                        visible_status='stopped', actual_cost=resources()), b'', {})
        results = []
        if action['ports']:
            results.append(self.run(action['ports'], action['sample_hz'], execution_index))
        if action['calibration_pair']:
            results.append(self.calibrate(action['calibration_pair'], burst_index))
        wire = b''.join(r.wire for r in results)
        obs = dict(action_key=action['action_key'], records=[x for r in results for x in r.observation['records']],
                   calibration_bursts=[x for r in results for x in r.observation['calibration_bursts']], visible_status='ok',
                   actual_cost=resources(int(bool(action['ports'])), len(wire), int(bool(action['calibration_pair'])), 1))
        if any(obs['actual_cost'][k] > action['cost_bound'][k] for k in obs['actual_cost']):
            raise RuntimeError('acquisition exceeded its public conservative bound')
        metrics = {str(i): r.metrics for i, r in enumerate(results)}
        return ExecutionResult(obs, wire, metrics)
