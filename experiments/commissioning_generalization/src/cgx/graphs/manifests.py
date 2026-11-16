"""Pre-training manifest construction. No outcome-dependent graph selection."""
from collections import Counter, defaultdict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import os
import time
import numpy as np
from .generator import generate_graph, graph_statistics, matched_variant, validate_graph, FAMILY
from .identity import IdentityIndex, structure_hash, clock_hash, isomorphic
from cgx.seeds import seed_for
from cgx.provenance import atomic_json, canonical_hash, file_hash


def read_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def load_cases(root, split):
    """Open exactly one split. Training never deserializes the sealed test labels."""
    if split not in ("train","validation","test_seen","test_connections","test_clocks","test_size","test_implementation"):
        raise ValueError("An explicit registered case split is required")
    for case in read_jsonl(Path(root)/"private"/"cases"/(split+".jsonl")):
        if case["split"] != split: raise ValueError("Private split file contains an alien case")
        yield case


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True, separators=(",",":"), allow_nan=False)+"\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def _clock_variant(graph, seed):
    rng = np.random.default_rng(seed)
    desired = graph_statistics(graph)
    fallback = None
    for _ in range(4000):
        out = deepcopy(graph)
        domains = [n["clock_domain"] for n in out["nodes"]]; rng.shuffle(domains)
        for n,d in zip(out["nodes"],domains): n["clock_domain"] = d
        if isomorphic(graph,out,clocks=True): continue
        fallback = out
        if graph_statistics(out)["cross_clock_edges"] == desired["cross_clock_edges"]: return out, []
    if fallback is None: raise ValueError("Cannot construct withheld clock partition")
    return fallback, ["cross_clock_edge_count_not_matched"]


def build_graph_records(config):
    """Return deterministic accepted graphs and rejection/matching bookkeeping."""
    if config['protocol_id']=='CGX-2.0':
        from .v2 import build_records
        return build_records(config,os.environ.get('CGX_REUSE_SOURCE'))
    index = IdentityIndex(); records = []; graphs = {}; rejected = Counter()
    seed = config["seeds"]["manifest"]
    def register(split, graph, i, anchor=None, deviations=None):
        cid = f"{split}-{i:03d}-{canonical_hash(graph)[:12]}"
        row = {"configuration_id":cid,"split":split,"structure_hash":structure_hash(graph),
               "clock_partition_hash":clock_hash(graph),"graph_hash":canonical_hash(graph),
               "implementation":"B" if split == "test_implementation" else "A",
               "source_fold":int(seed_for(config["seeds"]["crossfit_assignment"],structure_hash(graph)) % config["training"]["crossfit_folds"]) if split == "train" else None,
               "anchor_configuration_id":anchor,"statistics":graph_statistics(graph),"matching_deviations":deviations or []}
        records.append(row); graphs[cid] = graph
        return cid
    for split in ["train","validation"]:
        spec = config["splits"][split]; accepted = 0
        for trial in range(100000):
            if accepted == spec["configurations"]: break
            rng = np.random.default_rng(seed_for(seed,split,trial))
            size = int(rng.integers(spec["component_count"][0],spec["component_count"][1]+1))
            motif = ["chain","single","double"][trial % 3]
            try: graph = generate_graph(seed_for(seed,split,"graph",trial),size,motif,2+trial%2)
            except ValueError as exc:
                rejected[f"{split}:invalid_motif_size"] += 1; continue
            if index.contains(graph): rejected[f"{split}:duplicate_structure"] += 1; continue
            index.add(graph); register(split,graph,accepted); accepted += 1
        if accepted != spec["configurations"]: raise RuntimeError(f"Generator exhausted for {split}: {accepted}")
    # Anchor selection uses structural availability alone, before faults or outcomes exist.
    sources = sorted([r for r in records if r["split"]=="train"], key=lambda r: canonical_hash([seed,r["structure_hash"]]))
    count = config["splits"]["test_seen"]["configurations"]; anchors = []
    for src in sources:
        if len(anchors) == count: break
        base = graphs[src["configuration_id"]]; variant = None
        for trial in range(500):
            candidate = matched_variant(base,seed_for(seed,"matched",src["structure_hash"],trial))
            if not index.contains(candidate): variant = candidate; break
        if variant is None:
            rejected["anchor:no_unseen_reordering"] += 1; continue
        i = len(anchors); anchors.append(src["configuration_id"])
        register("test_seen",deepcopy(base),i,src["configuration_id"])
        index.add(variant); register("test_connections",variant,i,src["configuration_id"])
        clock, deviations = _clock_variant(base,seed_for(seed,"clock_holdout",src["structure_hash"]))
        register("test_clocks",clock,i,src["configuration_id"],deviations)
        register("test_implementation",deepcopy(base),i,src["configuration_id"])
    if len(anchors) != count: raise RuntimeError("Insufficient structurally matchable anchors")
    spec = config["splits"]["test_size"]; accepted = 0
    for trial in range(100000):
        if accepted == spec["configurations"]: break
        rng = np.random.default_rng(seed_for(seed,"size",trial))
        size = int(rng.integers(spec["component_count"][0],spec["component_count"][1]+1))
        graph = generate_graph(seed_for(seed,"size_graph",trial),size,["chain","single","double"][trial%3],2+trial%2)
        if index.contains(graph): rejected["test_size:duplicate_structure"] += 1; continue
        index.add(graph); register("test_size",graph,accepted); accepted += 1
    if accepted != spec["configurations"]: raise RuntimeError("Insufficient size holdout graphs")
    return records,graphs,{"rejections":dict(rejected),"anchors":anchors,"anchor_selection":"manifest-seeded hash order, first graphs with a nonisomorphic type-order variant"}


def _cases(config, records, graphs):
    reused=os.environ.get('CGX_REUSE_SOURCE') if config['protocol_id']=='CGX-2.0' else None
    if reused:
        from .v2 import selected_source_cases
        yield from selected_source_cases(config,records,reused)
    for row in records:
        split,cid = row["split"],row["configuration_id"]
        if reused and split in ('train','validation'):continue
        graph = graphs[cid]; count = config["splits"][split]["cases_per_configuration"]
        if count%4: raise ValueError("Family balancing requires cases divisible by four")
        for family in config["targets"]["families"]:
            legal = [n["id"] for n in graph["nodes"] if FAMILY.get(n["type"]) == family]
            rng = np.random.default_rng(seed_for(config["seeds"]["case_generation"],cid,family))
            locations = [legal[j%len(legal)] for j in range(count//4)] if legal else [None]*(count//4)
            rng.shuffle(locations)
            if family == "adapter_scale": severities = [s*m for m in config["faults"]["adapter_scale_magnitudes"] for s in config["faults"]["adapter_scale_signs"]]
            elif family == "cache_stale": severities = config["faults"]["cache_stale_delay_ticks"]
            elif family == "queue_hold": severities = config["faults"]["queue_hold_delay_ticks"]
            else: severities = [None]
            balanced = [severities[j%len(severities)] for j in range(count//4)]; rng.shuffle(balanced)
            for j,(location,severity) in enumerate(zip(locations,balanced)):
                case_id = canonical_hash([config["seeds"]["case_generation"],cid,family,j])
                streams = {name:seed_for(case_id,name) for name in ("task","noise","audit","verification","sensor","clock","recorder","timestamp","calibration","behavior")}
                clock_rng = np.random.default_rng(streams["clock"])
                offsets = dict(zip(graph["clock_domains"],map(float,clock_rng.uniform(*config["sensing"]["observer_offset_uniform_ms"],size=len(graph["clock_domains"])))))
                yield dict(case_id=case_id,split=split,configuration_id=cid,fault_family=family,fault_node=location,severity=severity,observer_offsets_ms=offsets,streams=streams,example_only=False)


def _qualify_one(args):
    graph,config,seed = args
    from cgx.env.system import System, generate_task
    case = dict(case_id="qualification",split="train",configuration_id="qualification",fault_family="healthy",fault_node=None,severity=None,
                observer_offsets_ms={d:0. for d in graph["clock_domains"]},streams={k:seed_for(seed,k) for k in ("task","noise","sensor","clock","recorder","timestamp","calibration","audit")},example_only=True)
    result = System(graph,case,generate_task(seed),config=config).run(execution_index=0)
    return result.metrics


def make_manifests(config, root, qualify=True, workers=1):
    root = Path(root)
    if (root/"protocol_lock.json").exists(): raise RuntimeError("Protocol is sealed; manifests cannot be overwritten")
    records,graphs,bookkeeping = build_graph_records(config)
    if not qualify: raise ValueError("Formal manifests require physical healthy qualification")
    seeds = [seed_for(config["seeds"]["healthy_qualification"],i) for i in range(3)]
    work = [(graphs[r["configuration_id"]],config,s) for r in records for s in seeds]
    start = time.monotonic()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool: results = list(pool.map(_qualify_one,work))
    else: results = [_qualify_one(x) for x in work]
    qualifications = []
    for i,row in enumerate(records):
        fixtures = results[3*i:3*i+3]
        passed = all(bool(x.get("passed",x.get("task_pass",False))) for x in fixtures)
        qualifications.append(dict(configuration_id=row["configuration_id"],state="passed" if passed else "failed",fixtures=fixtures))
    atomic_json(root/"reports"/"manifest_qualification.json",{"state":"passed" if all(x["state"]=="passed" for x in qualifications) else "failed","rows":qualifications,"elapsed_seconds":time.monotonic()-start,"criteria":"identical three source-seeded healthy task fixtures for every graph"})
    if any(x["state"]!="passed" for x in qualifications): raise RuntimeError("Healthy graph qualification failed; no manifests sealed; see manifest_qualification.json")
    cases = list(_cases(config,records,graphs))
    from cgx.env.system import generate_task
    write_jsonl(root/"manifests"/"split_manifest.jsonl",records)
    write_jsonl(root/"manifests"/"public_graphs.jsonl",({"configuration_id":cid,"graph":graph} for cid,graph in graphs.items()))
    write_jsonl(root/"private"/"private_cases.jsonl",cases)
    for split in config["splits"]:
        write_jsonl(root/"private"/"cases"/(split+".jsonl"),(c for c in cases if c["split"]==split))
    write_jsonl(root/"manifests"/"tasks.jsonl",({"case_id":c["case_id"],"task":generate_task(c["streams"]["task"])} for c in cases))
    repair = []
    for split in config["splits"]:
        if not split.startswith("test_"): continue
        chosen = sorted([r for r in records if r["split"]==split],key=lambda r:canonical_hash(["repair",r["configuration_id"]]))[:config["evaluation"]["repair_configurations_per_test_suite"]]
        for row in chosen:
            selected=[]
            for family in config["targets"]["families"]:
                selected += sorted([c["case_id"] for c in cases if c["configuration_id"]==row["configuration_id"] and c["fault_family"]==family],key=lambda x:canonical_hash(["repair",x]))[:config["evaluation"]["repair_cases_per_configuration"]//4]
            repair.append(dict(split=split,configuration_id=row["configuration_id"],case_ids=selected))
    atomic_json(root/"manifests"/"repair_subset.json",repair)
    # The same deterministic subset is used for costly horizon sensitivities.
    atomic_json(root/"manifests"/"round_sensitivity_subset.json",repair)
    atomic_json(root/"reports"/"graph_generation.json",bookkeeping)
    hashes={str(p.relative_to(root)):file_hash(p) for p in [root/"manifests"/"split_manifest.jsonl",root/"manifests"/"public_graphs.jsonl",root/"private"/"private_cases.jsonl",root/"manifests"/"tasks.jsonl",root/"manifests"/"repair_subset.json",root/"manifests"/"round_sensitivity_subset.json",*(root/"private"/"cases").glob("*.jsonl")]}
    atomic_json(root/"manifests"/"hashes.json",hashes)
    return validate_manifests(config,root,strict=True)


def validate_manifests(config, root, strict=True):
    root=Path(root); rows=list(read_jsonl(root/"manifests"/"split_manifest.jsonl"))
    graphs={r["configuration_id"]:r["graph"] for r in read_jsonl(root/"manifests"/"public_graphs.jsonl")}
    cases=list(read_jsonl(root/"private"/"private_cases.jsonl")); byid={r["configuration_id"]:r for r in rows}
    for split in config["splits"]:
        if list(load_cases(root,split)) != [c for c in cases if c["split"]==split]: raise ValueError("Private split projection differs from sealed case manifest")
    if len(byid)!=len(rows): raise ValueError("Duplicate configuration IDs")
    index=IdentityIndex(); source_index=IdentityIndex(); coverage=defaultdict(set)
    for split in config["splits"]:
        selected=[r for r in rows if r["split"]==split]
        if len(selected)!=config["splits"][split]["configurations"]: raise ValueError(f"Wrong graph count for {split}")
        for row in selected:
            graph=graphs[row["configuration_id"]]; validate_graph(graph)
            if canonical_hash(graph)!=row["graph_hash"]: raise ValueError("Graph content hash mismatch")
            n=len(graph["nodes"]); lo,hi=config["splits"][split]["component_count"]
            if not lo<=n<=hi: raise ValueError("Component count mismatch")
            if split in ("train","validation","test_connections","test_size"):
                if index.contains(graph): raise ValueError(f"Labeled-isomorphic graph leak in {split}")
                index.add(graph)
            if split=="train": source_index.add(graph)
            for edge in graph["edges"]:
                lookup={n["id"]:n for n in graph["nodes"]}
                coverage[split].add((lookup[edge["source"]]["type"],edge["relation"],lookup[edge["target"]]["type"]))
            if row["anchor_configuration_id"]:
                anchor=graphs[row["anchor_configuration_id"]]
                if split in ("test_seen","test_clocks","test_implementation") and not isomorphic(graph,anchor): raise ValueError("Anchor structure changed")
                if split=="test_clocks" and isomorphic(graph,anchor,True): raise ValueError("Clock partition not held out")
                if split=="test_connections":
                    left,right=graph_statistics(graph),graph_statistics(anchor)
                    if any(left[k]!=right[k] for k in ("components","type_counts","branch_count","degree_histogram")): raise ValueError("Unseen connections not matched")
    # Familiar edge types are guaranteed, not merely reported.
    absent=coverage["test_connections"]-coverage["train"]
    if absent: raise ValueError(f"Primary suite has unseen type relationships: {absent}")
    ids=set(); streams=set(); family_counts=Counter(); locations=defaultdict(Counter)
    for case in cases:
        cid=case["configuration_id"]; row=byid[cid]
        if case["case_id"] in ids: raise ValueError("Case overlap")
        ids.add(case["case_id"])
        for kind,value in case["streams"].items():
            if value in streams: raise ValueError("RNG stream overlap")
            streams.add(value)
        if case["split"]!=row["split"]: raise ValueError("Case split mismatch")
        family_counts[cid,case["fault_family"]]+=1
        if case["fault_family"]!="healthy":
            nodes={n["id"]:n for n in graphs[cid]["nodes"]}
            if FAMILY.get(nodes[case["fault_node"]]["type"])!=case["fault_family"]: raise ValueError("Illegal target")
            locations[cid,case["fault_family"]][case["fault_node"]]+=1
    for row in rows:
        cid=row["configuration_id"]; count=config["splits"][row["split"]]["cases_per_configuration"]//4
        if any(family_counts[cid,f]!=count for f in config["targets"]["families"]): raise ValueError("Family imbalance")
    for values in locations.values():
        if max(values.values())-min(values.values())>1: raise ValueError("Location imbalance")
    tasks=list(read_jsonl(root/"manifests"/"tasks.jsonl"))
    if {t["case_id"] for t in tasks}!=ids or len(tasks)!=len(ids): raise ValueError("Task manifest mismatch")
    hashes=json.loads((root/"manifests"/"hashes.json").read_text())
    for path,digest in hashes.items():
        if file_hash(root/path)!=digest: raise ValueError(f"Manifest changed: {path}")
    report={"state":"passed","graphs":len(rows),"cases":len(cases),"strict":strict,"hashes":hashes,
            "edge_type_coverage":{s:[list(t) for t in sorted(v)] for s,v in coverage.items()},"checks":["D01","D02","D06","D08","D09","S07","S08"]}
    atomic_json(root/"reports"/"manifest_validation.json",report)
    return report
