"""Seven-channel DAGs with real disjoint split/merge branches.

All transforms are canonical identity conversions in CGX-1.0 implementation v1.
Their declared scale and offset remain explicit, and faults alter executed payloads.
No random nominal parameter can manufacture a new topology identity.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import numpy as np
import networkx as nx

FAMILY = {"cache": "cache_stale", "queue": "queue_hold", "adapter": "adapter_scale"}


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], "little")


def legal_targets(graph):
    counts = Counter(n["type"] for n in graph["nodes"])
    keys, prior = ["healthy"], [.25]
    for node in graph["nodes"]:
        if node["type"] in FAMILY:
            keys.append(node["id"] + ":" + FAMILY[node["type"]])
            prior.append(.25 / counts[node["type"]])
    return keys, prior


def _port(name, direction, channels, side):
    return dict(name=name, direction=direction, width=len(channels) * (2 if side == "state" else 1),
                unit=side + "_rad" if side == "state" else "command_rad", channels=list(channels))


def generate_graph(seed, component_count=12, motif="chain", branches=2, clock_count=None):
    rng = np.random.default_rng(seed)
    if motif not in ("chain", "single", "double") or branches not in (2, 3):
        raise ValueError("Unknown motif/branch count")
    if not 8 <= component_count <= 32:
        raise ValueError("Physical component count outside protocol ranges")
    split_sides = [] if motif == "chain" else (["state", "command"] if motif == "double" else [str(rng.choice(["state", "command"]))])
    minimum = 3 + 2 * len(split_sides) + sum(branches if s in split_sides else 1 for s in ("state", "command"))
    if component_count < minimum:
        raise ValueError(f"{motif}/{branches} requires at least {minimum} components")
    modules = component_count - 3 - 2 * len(split_sides)
    paths = [(side, b) for side in ("state", "command") for b in range(branches if side in split_sides else 1)]
    lengths = [1] * len(paths)
    for _ in range(modules - len(paths)):
        lengths[int(rng.integers(len(paths)))] += 1
    types = {}
    for (side, b), length in zip(paths, lengths):
        types[(side, b)] = list(rng.choice(["cache" if side == "state" else "queue", "adapter"], size=length))
    # All three mechanisms have actual executable locations, even in small graphs.
    state_slots = [(p, k) for p in paths if p[0] == "state" for k in range(len(types[p]))]
    command_slots = [(p, k) for p in paths if p[0] == "command" for k in range(len(types[p]))]
    s = state_slots[int(rng.integers(len(state_slots)))]; types[s[0]][s[1]] = "cache"
    c = command_slots[int(rng.integers(len(command_slots)))]; types[c[0]][c[1]] = "queue"
    others = [x for x in state_slots + command_slots if x != s and x != c]
    a = others[int(rng.integers(len(others)))]; types[a[0]][a[1]] = "adapter"
    nodes, edges = [], []
    def add(kind, ports):
        nid = "v" + str(len(nodes))
        nominal = ({"scale": 1., "offset": 0.} if kind == "adapter" else
                   {"state_age_ticks": 0} if kind == "cache" else
                   {"queue_delay_ticks": 0} if kind == "queue" else {})
        nodes.append(dict(id=nid, type=kind, ports=ports, clock_domain="", nominal=nominal))
        return nid
    def edge(src, sp, dst, dp, side):
        edges.append(dict(source=src, source_port=sp, target=dst, target_port=dp, relation=side+"_flow"))
    channels = list(range(7))
    sensor = add("sensor", [_port("out", "out", channels, "state")])
    controller = add("controller", [_port("in", "in", channels, "state"), _port("out", "out", channels, "command")])
    actuator = add("actuator", [_port("in", "in", channels, "command")])
    for side, source, target in [("state", sensor, controller), ("command", controller, actuator)]:
        if side in split_sides:
            # Channel partition is physical, and reused when generating matched variants.
            partitions = [list(map(int, q)) for q in np.array_split(np.arange(7), branches)]
            splitter = add("splitter", [_port("in", "in", channels, side)] + [_port(f"out{i}", "out", q, side) for i,q in enumerate(partitions)])
            merger = add("merger", [_port(f"in{i}", "in", q, side) for i,q in enumerate(partitions)] + [_port("out", "out", channels, side)])
            edge(source, "out", splitter, "in", side); edge(merger, "out", target, "in", side)
        else:
            partitions = [channels]; splitter, merger = source, target
        for b, channel_subset in enumerate(partitions):
            previous, previous_port = splitter, f"out{b}" if side in split_sides else "out"
            for kind in types[(side,b)]:
                current = add(str(kind), [_port("in", "in", channel_subset, side), _port("out", "out", channel_subset, side)])
                edge(previous, previous_port, current, "in", side)
                previous, previous_port = current, "out"
            edge(previous, previous_port, merger, f"in{b}" if side in split_sides else "in", side)
    clock_count = int(clock_count or rng.integers(3 if component_count > 16 else 2, (8 if component_count > 16 else 5)+1))
    memberships = np.arange(component_count) % clock_count
    rng.shuffle(memberships)
    for node, d in zip(nodes, memberships): node["clock_domain"] = f"d{d}"
    # Require both kinds of timing comparison, via the graph's executable links.
    lookup = {n["id"]: n for n in nodes}
    if not any(lookup[e["source"]]["clock_domain"] == lookup[e["target"]]["clock_domain"] for e in edges):
        lookup[edges[0]["target"]]["clock_domain"] = lookup[edges[0]["source"]]["clock_domain"]
    used = sorted({n["clock_domain"] for n in nodes})
    graph = dict(schema_version="cgx.graph.v1", nodes=nodes, edges=edges, clock_domains=used, sample_rates_hz=[10,50,100])
    graph, _ = relabel_graph(graph, _seed(seed, "relabel"))
    validate_graph(graph)
    return graph


def relabel_graph(graph, seed):
    rng = np.random.default_rng(seed)
    out = deepcopy(graph)
    names = rng.permutation(len(out["nodes"]))
    node_map = {n["id"]: f"n{int(v)}" for n,v in zip(out["nodes"], names)}
    clock_names = rng.permutation(len(out["clock_domains"]))
    clock_map = {d:f"c{int(v)}" for d,v in zip(out["clock_domains"], clock_names)}
    for n in out["nodes"]:
        n["id"] = node_map[n["id"]]; n["clock_domain"] = clock_map[n["clock_domain"]]
    rng.shuffle(out["nodes"])
    for e in out["edges"]: e["source"],e["target"] = node_map[e["source"]],node_map[e["target"]]
    rng.shuffle(out["edges"])
    out["clock_domains"] = [clock_map[d] for d in out["clock_domains"]]
    return out, node_map


def graph_statistics(graph):
    counts = Counter(n["type"] for n in graph["nodes"])
    directed = nx.DiGraph((e["source"],e["target"]) for e in graph["edges"])
    lookup = {n["id"]:n for n in graph["nodes"]}
    return {"components":len(graph["nodes"]), "type_counts":dict(sorted(counts.items())),
            "branch_count":sum(max(0,directed.out_degree(n)-1) for n in directed),
            "degree_histogram": sorted([list(x) for x in Counter((directed.in_degree(n),directed.out_degree(n)) for n in directed).items()]),
            "clock_count":len(graph["clock_domains"]),
            "clock_sizes":sorted(Counter(n["clock_domain"] for n in graph["nodes"]).values()),
            "cross_clock_edges":sum(lookup[e["source"]]["clock_domain"] != lookup[e["target"]]["clock_domain"] for e in graph["edges"])}


def matched_variant(graph, seed):
    """Reorder executable module types; preserve physical graph degree and counts."""
    out = deepcopy(graph); rng = np.random.default_rng(seed)
    for side in ("state", "command"):
        eligible = [n for n in out["nodes"] if n["type"] in FAMILY and n["ports"][0]["unit"] == side+"_rad"]
        values = [(n["type"],n["nominal"]) for n in eligible]
        rng.shuffle(values)
        for n, (kind, nominal) in zip(eligible,values): n["type"],n["nominal"] = kind,nominal
    # Preserve the anchor's observer partition for the primary connection contrast.
    # Clock rewiring belongs to its separate preregistered suite. This also preserves
    # both within-domain and cross-domain timing comparisons and their counts.
    validate_graph(out)
    return out


def validate_graph(graph):
    allowed = {"schema_version","nodes","edges","clock_domains","sample_rates_hz"}
    if set(graph) != allowed: raise ValueError("Unexpected/missing public graph fields")
    nodes = {n["id"]:n for n in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]): raise ValueError("Repeated component ID")
    types = Counter(n["type"] for n in nodes.values())
    if any(types[t] != 1 for t in ("sensor","controller","actuator")): raise ValueError("Need unique end components")
    if any(types[t] < 1 for t in FAMILY): raise ValueError("Missing applicable fault family")
    dag = nx.DiGraph(); dag.add_nodes_from(nodes)
    ports = {(n["id"],p["name"]):p for n in nodes.values() for p in n["ports"]}
    incoming, outgoing = Counter(), Counter()
    for e in graph["edges"]:
        s,t = (e["source"],e["source_port"]),(e["target"],e["target_port"])
        if s not in ports or t not in ports: raise ValueError("Unknown edge endpoint")
        sp,tp = ports[s],ports[t]
        if sp["direction"] != "out" or tp["direction"] != "in": raise ValueError("Edge direction")
        if any(sp[k] != tp[k] for k in ("width","unit","channels")): raise ValueError("Payload contract mismatch")
        if e["relation"] != ("state_flow" if sp["unit"] == "state_rad" else "command_flow"): raise ValueError("Relation mismatch")
        outgoing[s] += 1; incoming[t] += 1; dag.add_edge(e["source"],e["target"])
    if not nx.is_directed_acyclic_graph(dag): raise ValueError("Software dataflow must be DAG")
    for key,p in ports.items():
        if (incoming if p["direction"] == "in" else outgoing)[key] != 1: raise ValueError("Every port must route exactly once")
    for n in nodes.values():
        if n["clock_domain"] not in graph["clock_domains"]: raise ValueError("Unknown clock")
        if n["type"] in ("splitter","merger"):
            side = "out" if n["type"] == "splitter" else "in"
            partition = [j for p in n["ports"] if p["direction"] == side for j in p["channels"]]
            full = [j for p in n["ports"] if p["direction"] != side for j in p["channels"]]
            if sorted(partition) != sorted(full) or len(set(partition)) != len(partition): raise ValueError("Bad channel partition")
        if n["type"] == "cache" and n["ports"][0]["unit"] != "state_rad": raise ValueError("Cache off state path")
        if n["type"] == "queue" and n["ports"][0]["unit"] != "command_rad": raise ValueError("Queue off command path")
    cross=[nodes[e["source"]]["clock_domain"]!=nodes[e["target"]]["clock_domain"] for e in graph["edges"]]
    if not any(cross) or all(cross): raise ValueError("Need both within-domain and cross-domain timing pairs")
    sensor = next(n["id"] for n in nodes.values() if n["type"] == "sensor")
    actuator = next(n["id"] for n in nodes.values() if n["type"] == "actuator")
    if any(not nx.has_path(dag,sensor,n) or not nx.has_path(dag,n,actuator) for n in dag): raise ValueError("Decorative/unreachable component")
    return True
