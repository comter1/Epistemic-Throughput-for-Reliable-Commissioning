"""Public graph-derived acquisition catalog and exact pathwise resource accounting."""
from __future__ import annotations
import hashlib
import json
import math
from .protocol import require_supported

RESOURCE_KEYS = ('executions', 'bytes', 'calibrations', 'decisions')

def resources(executions=0, bytes=0, calibrations=0, decisions=0):
    values = dict(executions=executions, bytes=bytes, calibrations=calibrations, decisions=decisions)
    if any(type(v) is not int or v < 0 for v in values.values()):
        raise ValueError('resources must be nonnegative integers')
    return values

def trace_ports(graph):
    return [f"{n['id']}:{p['name']}" for n in graph['nodes'] for p in n['ports']]

def full_run_bytes(graph, config=None):
    require_supported(config)
    return 1000 * 128 * len(trace_ports(graph)) + 100 * 128

def initial_budget(graph, fraction, execution_cap=4, calibration_cap=2):
    if fraction < 0:
        raise ValueError('negative byte fraction')
    return resources(execution_cap, math.floor(fraction * 4 * full_run_bytes(graph)), calibration_cap,
                     execution_cap + calibration_cap)

def can_afford(remaining, cost):
    return all(remaining[k] >= cost[k] for k in RESOURCE_KEYS)

def charge(remaining, cost):
    resources(**cost)
    if not can_afford(remaining, cost):
        raise ValueError('pathwise resource cap exceeded')
    return {k: remaining[k] - cost[k] for k in RESOURCE_KEYS}

def feasible_actions(actions, remaining):
    return [a for a in actions if can_afford(remaining, a['cost_bound'])]

def action_catalog(graph, config=None):
    """No case-specific argument is accepted. Bounds precede any execution."""
    require_supported(config)
    ports = trace_ports(graph)
    clocks = {f"{n['id']}:{p['name']}": n['clock_domain'] for n in graph['nodes'] for p in n['ports']}
    catalog, seen = [], set()
    def add(kind, selected=(), rate=0, pair=None):
        selected = tuple(sorted(set(selected)))
        pair = tuple(sorted(pair)) if pair else None
        identity = (selected, rate, pair, kind == 'stop')
        if identity in seen:
            return
        seen.add(identity)
        execution = int(bool(selected))
        calibration = int(pair is not None)
        bound = 128 * 10 * rate * len(selected) + execution * 12800 + calibration * 160
        action_key = 'stop' if kind == 'stop' else hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
        catalog.append(dict(action_key=action_key, kind=kind, ports=list(selected), sample_hz=rate,
                            calibration_pair=list(pair) if pair else None,
                            cost_bound=resources(execution, bound, calibration, int(kind != 'stop'))))
    pairs = []
    for node in graph['nodes']:
        if node['type'] in ('cache', 'queue', 'adapter'):
            selected = [f"{node['id']}:{p['name']}" for p in node['ports']]
            pairs.append(('module_pair', selected))
    for edge in graph['edges']:
        pairs.append(('link_pair', [f"{edge['source']}:{edge['source_port']}",
                                    f"{edge['target']}:{edge['target_port']}"]))
    calibration_pairs = set()
    for rate in graph.get('sample_rates_hz', [10, 50, 100]):
        for port in ports:
            add('single_port', [port], rate)
        for kind, selected in pairs:
            add(kind, selected, rate)
            domains = sorted(set(clocks[p] for p in selected))
            if len(domains) == 2:
                add('pair_plus_calibration', selected, rate, domains)
                calibration_pairs.add(tuple(domains))
        add('full_trace', ports, rate)
    for pair in sorted(calibration_pairs):
        add('calibration_only', pair=pair)
    add('stop')
    return catalog
