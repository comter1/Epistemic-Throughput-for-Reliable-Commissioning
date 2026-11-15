"""Identity excludes arbitrary names and continuous nominal parameter draws."""
import hashlib
import json
import networkx as nx


def expanded_graph(graph, clocks=False):
    out = nx.DiGraph()
    for node in graph["nodes"]:
        nid = ("node", node["id"])
        out.add_node(nid, label="component:" + node["type"])
        for port in node["ports"]:
            pid = ("port", node["id"], port["name"])
            # Port spelling is arbitrary. The physical channel/interface role is not.
            label = json.dumps([port["direction"], port["width"], port["unit"], sorted(port["channels"])])
            out.add_node(pid, label="port:" + label)
            out.add_edge(nid, pid, label="owns")
        if clocks:
            cid = ("clock", node["clock_domain"])
            out.add_node(cid, label="clock")
            out.add_edge(nid, cid, label="membership")
    for edge in graph["edges"]:
        out.add_edge(("port", edge["source"], edge["source_port"]),
                     ("port", edge["target"], edge["target_port"]), label=edge["relation"])
    return out


def structure_hash(graph, clocks=False):
    return nx.weisfeiler_lehman_graph_hash(expanded_graph(graph, clocks), node_attr="label", edge_attr="label", iterations=5)


def clock_hash(graph):
    return structure_hash(graph, clocks=True)


def isomorphic(left, right, clocks=False):
    return nx.is_isomorphic(expanded_graph(left, clocks), expanded_graph(right, clocks),
                            node_match=nx.algorithms.isomorphism.categorical_node_match("label", ""),
                            edge_match=nx.algorithms.isomorphism.categorical_edge_match("label", ""))


class IdentityIndex:
    """WL is only a filter; a collision ALWAYS receives an exact graph match."""
    def __init__(self, clocks=False):
        self.clocks = clocks
        self.buckets = {}

    def contains(self, graph):
        return any(isomorphic(graph, other, self.clocks)
                   for other in self.buckets.get(structure_hash(graph, self.clocks), []))

    def add(self, graph):
        self.buckets.setdefault(structure_hash(graph, self.clocks), []).append(graph)
