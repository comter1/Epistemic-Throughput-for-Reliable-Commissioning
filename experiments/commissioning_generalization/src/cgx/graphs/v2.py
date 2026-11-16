"""V2 topology holdouts and outcome-independent projection of compatible source cases."""
from collections import defaultdict,Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from cgx.provenance import canonical_hash
from cgx.seeds import seed_for
from .generator import generate_graph,graph_statistics,matched_variant
from .identity import IdentityIndex,structure_hash,clock_hash


def selected_source_cases(config,records,source_root):
    from .manifests import load_cases
    wanted={r['configuration_id']:r for r in records if r['split'] in ('train','validation')}
    groups=defaultdict(list)
    for split in ('train','validation'):
        for case in load_cases(source_root,split):
            if case['configuration_id'] in wanted:groups[case['configuration_id'],case['fault_family']].append(case)
    selected=[]
    for cid,row in wanted.items():
        count=config['splits'][row['split']]['cases_per_configuration']//4
        for family in config['targets']['families']:
            locations=defaultdict(list)
            for case in groups[cid,family]:locations[str(case['fault_node'])].append(case)
            for values in locations.values():values.sort(key=lambda c:canonical_hash(['v2-case-selection',c['case_id']]))
            keys=sorted(locations);chosen=[]
            if not keys:raise ValueError('Source projection missing a family')
            while len(chosen)<count:
                for key in keys:
                    if len(chosen)==count:break
                    if not locations[key]:raise ValueError('Insufficient balanced source cases')
                    chosen.append(locations[key].pop(0))
            selected.extend(chosen)
    return selected


def build_records(config,source_root=None):
    from .manifests import read_jsonl
    records=[];graphs={};index=IdentityIndex();seed=config['seeds']['manifest'];rejected=Counter()
    def register(split,graph,i,anchor=None):
        cid=f'{split}-{i:03d}-{canonical_hash(graph)[:12]}'
        row=dict(configuration_id=cid,split=split,structure_hash=structure_hash(graph),clock_partition_hash=clock_hash(graph),
            graph_hash=canonical_hash(graph),implementation='A',source_fold=int(seed_for(config['seeds']['crossfit_assignment'],structure_hash(graph))%3) if split=='train' else None,
            anchor_configuration_id=anchor,statistics=graph_statistics(graph),matching_deviations=[])
        records.append(row);graphs[cid]=graph;return cid
    originals=list(read_jsonl(Path(source_root)/'manifests/split_manifest.jsonl')) if source_root else []
    original_graphs={r['configuration_id']:r['graph'] for r in read_jsonl(Path(source_root)/'manifests/public_graphs.jsonl')} if source_root else {}
    for split in ('train','validation'):
        spec=config['splits'][split];accepted=0
        if source_root:
            for row in [r for r in originals if r['split']==split][:spec['configurations']]:
                graph=original_graphs[row['configuration_id']]
                if index.contains(graph):raise ValueError('Original source projection leaks topology')
                index.add(graph);records.append(deepcopy(row));graphs[row['configuration_id']]=deepcopy(graph);accepted+=1
        else:
            for trial in range(100000):
                if accepted==spec['configurations']:break
                rng=np.random.default_rng(seed_for(seed,split,trial));size=int(rng.integers(spec['component_count'][0],spec['component_count'][1]+1))
                try:graph=generate_graph(seed_for(seed,split,'graph',trial),size,['chain','single','double'][trial%3],2+trial%2)
                except ValueError:continue
                if index.contains(graph):continue
                index.add(graph);register(split,graph,accepted);accepted+=1
        if accepted!=spec['configurations']:raise ValueError('Insufficient source topologies')
    sources=sorted([r for r in records if r['split']=='train'],key=lambda r:canonical_hash([seed,r['structure_hash']]))
    def relations(graph):
        nodes={n['id']:n for n in graph['nodes']}
        return {(nodes[e['source']]['type'],e['relation'],nodes[e['target']]['type']) for e in graph['edges']}
    source_relations=set().union(*(relations(graphs[r['configuration_id']]) for r in sources))
    seen_count=config['splits']['test_seen']['configurations'];unseen_count=config['splits']['test_connections']['configurations']
    if unseen_count%seen_count:raise ValueError('Primary topology matching requires an integer variants-per-anchor ratio')
    per_anchor=unseen_count//seen_count;anchors=[]
    for source in sources:
        if len(anchors)==seen_count:break
        base=graphs[source['configuration_id']];variants=[];local_index=IdentityIndex()
        for trial in range(2000):
            candidate=matched_variant(base,seed_for(seed,'v2-matched',source['structure_hash'],trial))
            if index.contains(candidate) or local_index.contains(candidate) or not relations(candidate)<=source_relations:continue
            local_index.add(candidate);variants.append(candidate)
            if len(variants)==per_anchor:break
        if len(variants)!=per_anchor:rejected['anchor_insufficient_distinct_variants']+=1;continue
        i=len(anchors);anchors.append(source['configuration_id']);register('test_seen',deepcopy(base),i,source['configuration_id'])
        for j,candidate in enumerate(variants):
            index.add(candidate);register('test_connections',candidate,i*per_anchor+j,source['configuration_id'])
    if len(anchors)!=seen_count:raise ValueError('Insufficient matched unseen topology anchors')
    spec=config['splits']['test_size'];accepted=0
    for trial in range(100000):
        if accepted==spec['configurations']:break
        rng=np.random.default_rng(seed_for(seed,'v2-size',trial));size=int(rng.integers(spec['component_count'][0],spec['component_count'][1]+1))
        clock_lo,clock_hi=config['system']['clock_domains_size']
        graph=generate_graph(seed_for(seed,'v2-size-graph',trial),size,['chain','single','double'][trial%3],
            branches=2+trial%2,clock_count=clock_lo+trial%(clock_hi-clock_lo+1))
        if index.contains(graph):continue
        index.add(graph);register('test_size',graph,accepted);accepted+=1
    if accepted!=spec['configurations']:raise ValueError('Insufficient larger held-out topologies')
    return records,graphs,dict(rejections=dict(rejected),anchors=anchors,variants_per_seen_anchor=per_anchor,
        source_projection=str(source_root) if source_root else None,selection='manifest-seeded structure/hash order; never simulator outcomes')
