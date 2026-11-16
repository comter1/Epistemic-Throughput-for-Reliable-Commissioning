"""Fixed acquired-only preprocessing. IDs are used for equality/lookup, never numerically.

The 64-wide deterministic bins precede the trainable event embedding and GRU.
This module deliberately accepts no environment/evaluator handle.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np
import torch

TYPES = ('sensor','cache','adapter','controller','queue','actuator','splitter','merger','clock')
FAMILIES = {'cache':'cache_stale','queue':'queue_hold','adapter':'adapter_scale'}
KINDS = ('single_port','module_pair','link_pair','pair_plus_calibration','calibration_only','full_trace','stop')
RESOURCES = ('executions','bytes','calibrations','decisions')
FEATURE_VERSION = 'cgx.acquired.v2.64bin'
NODE_WIDTH = 32
ACTION_WIDTH = 24


def action_dict(action):
    if isinstance(action, dict):
        return action
    from dataclasses import asdict
    a=asdict(action)
    a['action_key']=a.pop('key',a.get('action_key'))
    a['cost_bound']=a.pop('upper_cost',a.get('cost_bound'))
    return a


def target_space(graph):
    keys=['healthy']; indices=[-1]; families=[0]; prior=[.25]
    counts={t:sum(n['type']==t for n in graph['nodes']) for t in FAMILIES}
    for i,n in enumerate(graph['nodes']):
        if n['type'] in FAMILIES:
            keys.append(n['id']+':'+FAMILIES[n['type']]); indices.append(i)
            families.append(('cache','queue','adapter').index(n['type'])+1)
            prior.append(.25/counts[n['type']])
    if not all(counts.values()):
        raise ValueError('Graph lacks an applicable location for a protocol family')
    return keys,indices,families,prior


def _bin_sequence(records, matches, direction, cap=256):
    """No record ordinal, ID value or common-time marker is encoded."""
    rows=[]; previous=None
    start=float(records[0]['local_time_s'])
    for r in records:
        values=np.zeros(14,dtype=np.float32)
        values[:len(r['values'])]=r['values']
        token=r['message_token']; time=float(r['local_time_s'])
        peers=matches.get(token,())
        other=[p for p in peers if p['port']!=r['port']]
        delta=np.asarray([p['local_time_s']-time for p in other],dtype=np.float32)
        rows.append((values,time-start,0.0 if previous is None else time-previous['local_time_s'],
                     float(previous is not None and previous['message_token']==token),
                     float(r['missing']),delta,float(bool(other)),
                     ('receive','send','sample').index(r['event_type'])))
        previous=r
    # np.array_split produces at most two contiguous bin widths. Batch each
    # width so the original per-bin reductions keep their order and precision.
    # Peer timing still uses only acquired equality matches, exactly as above.
    count=min(cap,len(rows));small,large_count=divmod(len(rows),count)
    values=np.stack([row[0] for row in rows])
    scalars=np.asarray([[row[1],row[2],row[3],row[4],row[6],row[7]] for row in rows],dtype=np.float64)
    bins=np.zeros((count,64),np.float32);cursor=offset=0
    for number,width in ((large_count,small+1),(count-large_count,small)):
        if not number:continue
        end=offset+number*width
        vs=values[offset:end].reshape(number,width,14)
        ss=scalars[offset:end].reshape(number,width,6)
        v=bins[cursor:cursor+number]
        v[:,:14]=vs.mean(1);v[:,14:28]=vs.min(1);v[:,28:42]=vs.max(1)
        v[:,42:49]=vs[:,0,:7];v[:,49:53]=vs[:,-1,:4];v[:,53]=direction
        v[:,54]=(ss[:,:,5]==0).mean(1);v[:,55]=(ss[:,:,5]==1).mean(1)
        v[:,56]=ss[:,0,0];v[:,57]=ss[:,-1,0]
        v[:,58:61]=ss[:,:,1:4].mean(1);v[:,62]=ss[:,:,4].mean(1);v[:,63]=math.log1p(width)
        for j in range(number):
            begin=offset+j*width;deltas=[d for row in rows[begin:begin+width] for d in row[5]]
            v[j,61]=float(np.mean(deltas)) if deltas else 0.
        offset=end;cursor+=number
    return bins


@dataclass
class GraphFeatures:
    nodes: np.ndarray
    edge_index: np.ndarray
    edge_types: np.ndarray
    edge_features: np.ndarray
    sequences: list
    sequence_owners: list
    targets: list
    target_nodes: list
    target_families: list
    prior: list
    physical_count: int
    node_lookup: dict
    clock_lookup: dict
    resources: np.ndarray
    task: np.ndarray


def tensorize_history(history, device=None, prediction=None):
    """Return immutable numerical features assembled from explicit public fields only.

    `device` is accepted for the inference API; transfer occurs in collate_features.
    Extra bookkeeping keys are ignored. Fault fields cannot affect the result.
    """
    g=history['graph']; nn=len(g['nodes']); clocks=list(g['clock_domains'])
    lookup={n['id']:i for i,n in enumerate(g['nodes'])}
    clook={c:nn+i for i,c in enumerate(clocks)}
    nodes=np.zeros((nn+len(clocks),NODE_WIDTH),np.float32)
    for i,n in enumerate(g['nodes']):
        nodes[i,TYPES.index(n['type'])]=1
        ports=n['ports']; nom=n['nominal']
        nodes[i,9]=len(ports)/4
        nodes[i,10]=sum(p['width'] for p in ports if p['direction']=='in')/14
        nodes[i,11]=sum(p['width'] for p in ports if p['direction']=='out')/14
        for p in ports:
            nodes[i,12:19][p['channels']]=1
        nodes[i,19:23]=[nom.get('scale',1.),nom.get('offset',0.),nom.get('queue_delay_ticks',0)/10,nom.get('state_age_ticks',0)/10]
    for c,i in clook.items(): nodes[i,8]=1
    port_directions={n['id']+':'+p['name']:(-1 if p['direction']=='in' else 1) for n in g['nodes'] for p in n['ports']}
    seqs=[]; owners=[]; pair_values=defaultdict(list); cal_values=defaultdict(list)
    for observation in history.get('observations',()):
        records=observation.get('records',())
        grouped=defaultdict(list); matches=defaultdict(list)
        for r in records:
            grouped[r['port']].append(r); matches[r['message_token']].append(r)
        for port,rs in grouped.items():
            # Preserve emitted ordering: sorting on a noisy timestamp changes causality.
            seqs.append(_bin_sequence(rs,matches,port_directions.get(port,1)))
            owner=lookup.get(port.split(':')[0],-1)
            owners.append(owner)
            if owner>=0:
                nodes[owner,23]+=1; nodes[owner,24]+=len(rs)/1000
                nodes[owner,25]+=float(any(r['missing'] for r in rs))
            for r in rs:
                for p in matches[r['message_token']]:
                    if p['port']==port: continue
                    other=lookup.get(p['port'].split(':')[0],-1)
                    if owner>=0 and other>=0:
                        pair_values[(owner,other)].append(float(p['local_time_s']-r['local_time_s']))
        for burst in observation.get('calibration_bursts',()):
            a,b=burst['pair']; ts=np.asarray(burst['timestamps_s'])
            estimate=((ts[:,1]-ts[:,0])+(ts[:,2]-ts[:,3]))/2
            if a in clook and b in clook:
                cal_values[(clook[a],clook[b])].extend(estimate.tolist())
                cal_values[(clook[b],clook[a])].extend((-estimate).tolist())
    # Planning state holds only previously chosen public actions, not their outcomes.
    for a in history.get('planned_actions',()):
        for p in action_dict(a)['ports']:
            if p.split(':')[0] in lookup: nodes[lookup[p.split(':')[0]],26]+=1
    keys,target_nodes,families,prior=target_space(g)
    pred=prediction if prediction is not None else history.get('prediction')
    if pred is not None:
        probs=dict(zip(pred['target_keys'],np.exp2(pred['log_probabilities_base2'])))
        for key,i in zip(keys[1:],target_nodes[1:]): nodes[i,27]=probs.get(key,0.)
        nodes[:,28]=probs.get('healthy',0.)
    nodes[:,23:27]=np.log1p(nodes[:,23:27])
    nodes[:nn,29]=1 # physical vs representational
    nodes[:nn,30]=np.log1p(nn)
    nodes[:nn,31]=1-np.minimum(nodes[:nn,23],1) # explicitly unobserved
    edges=[]; et=[]; ef=[]
    def edge(a,b,typ,values=()):
        edges.append((a,b)); et.append(typ)
        vals=np.zeros(4,np.float32)
        if values: vals[:3]=[1,np.mean(values),np.std(values)]
        vals[3]=float(a<nn and b<nn and g['nodes'][a]['clock_domain']==g['nodes'][b]['clock_domain'])
        ef.append(vals)
    for e in g['edges']:
        a,b=lookup[e['source']],lookup[e['target']]
        typ=0 if e['relation']=='state_flow' else 2
        edge(a,b,typ,pair_values.get((a,b),()))
        edge(b,a,typ+1,pair_values.get((b,a),()))
    for n in g['nodes']:
        a,b=lookup[n['id']],clook[n['clock_domain']]
        edge(a,b,4); edge(b,a,5)
    for (a,b),values in cal_values.items(): edge(a,b,6,values)
    initial=history.get('initial_budget',{}); remaining=history.get('remaining',{})
    resources=np.asarray([remaining.get(k,0)/max(1,initial.get(k,0)) for k in RESOURCES]+
                         [remaining.get('executions',0)/8,remaining.get('calibrations',0)/2,
                          remaining.get('decisions',0)/10,math.log1p(initial.get('bytes',0))/20],np.float32)
    task_data=history.get('nominal_task',{}); requirements=history.get('requirements',{})
    task=np.zeros(34,np.float32)
    task[:7]=np.asarray(task_data.get('home',[0]*7))
    task[7:28]=np.asarray(task_data.get('waypoints',[[0]*7]*3)).reshape(-1)
    task[28:31]=np.asarray(task_data.get('deadlines_s',[0]*3))/10
    task[31:]=[requirements.get('position_tolerance_m',.02),requirements.get('hold_seconds',.5),requirements.get('joint_limit_tolerance_rad',.001)]
    return GraphFeatures(nodes,np.asarray(edges,np.int64).T,np.asarray(et,np.int64),np.asarray(ef,np.float32),
                         seqs,owners,keys,target_nodes,families,prior,nn,lookup,clook,resources,task)


def collate_features(features, device):
    """Disjoint graph batch with variable target, port and candidate cardinalities."""
    sizes=[len(f.nodes) for f in features]; offsets=np.cumsum([0]+sizes[:-1]); B=len(features)
    allseq=[s for f in features for s in f.sequences]
    max_t=max([len(s) for s in allseq]+[1]); events=np.zeros((max(1,len(allseq)),max_t,64),np.float32)
    lengths=[]; owners=[]; seqbatch=[]
    j=0
    for b,(f,off) in enumerate(zip(features,offsets)):
        for s,o in zip(f.sequences,f.sequence_owners):
            events[j,:len(s)]=s; lengths.append(len(s)); owners.append(-1 if o<0 else off+o); seqbatch.append(b); j+=1
    if not allseq: lengths=[1]; owners=[-1]; seqbatch=[0]
    max_targets=max(len(f.targets) for f in features)
    targets=np.full((B,max_targets),-1,np.int64); fam=np.zeros_like(targets); valid=np.zeros_like(targets,dtype=bool)
    for b,(f,off) in enumerate(zip(features,offsets)):
        targets[b,:len(f.targets)]=[off+i if i>=0 else -1 for i in f.target_nodes]
        fam[b,:len(f.targets)]=f.target_families; valid[b,:len(f.targets)]=True
    def tensor(v,dtype=None): return torch.as_tensor(v,dtype=dtype,device=device)
    return dict(nodes=tensor(np.concatenate([f.nodes for f in features])),
                edge_index=tensor(np.concatenate([f.edge_index+off for f,off in zip(features,offsets)],axis=1)),
                edge_types=tensor(np.concatenate([f.edge_types for f in features])),
                edge_features=tensor(np.concatenate([f.edge_features for f in features])),
                node_batch=tensor(np.repeat(np.arange(B),sizes)),sizes=sizes,offsets=offsets.tolist(),
                events=tensor(events),lengths=torch.tensor(lengths),sequence_owners=tensor(owners),sequence_batch=tensor(seqbatch),sequence_valid=tensor([bool(allseq)]*len(seqbatch)),
                target_nodes=tensor(targets),target_families=tensor(fam),target_valid=tensor(valid),
                physical_counts=tensor([f.physical_count for f in features]),resources=tensor(np.stack([f.resources for f in features])),
                task=tensor(np.stack([f.task for f in features])),features=features,batch_size=B)


def collate_actions(batch, action_sets, device):
    owners=[]; pool_action=[]; feats=[]; a_batch=[]; stop=[]; counts=[]
    for b,(f,off,actions) in enumerate(zip(batch['features'],batch['offsets'],action_sets)):
        counts.append(len(actions))
        for a in map(action_dict,actions):
            j=len(feats); ids=[]
            for p in a['ports']:
                name=p.split(':')[0]
                if name in f.node_lookup: ids.append(off+f.node_lookup[name])
            if a.get('calibration_pair'):
                ids.extend(off+f.clock_lookup[c] for c in a['calibration_pair'])
            for i in ids: owners.append(i); pool_action.append(j)
            v=np.zeros(ACTION_WIDTH,np.float32); v[KINDS.index(a['kind'])]=1
            v[7]=a['sample_hz']/100; v[8]=math.log1p(len(a['ports']))
            cost=a['cost_bound']; v[9:13]=[cost.get('executions',0)/8,math.log1p(cost.get('bytes',0))/20,cost.get('calibrations',0)/2,cost.get('decisions',0)/10]
            v[13:21]=f.resources
            # Relative candidate cost can distinguish affordable fractional bundles.
            v[21]=float(bool(a.get('calibration_pair'))); v[22]=len(ids)/max(1,len(f.nodes)); v[23]=1
            feats.append(v); a_batch.append(b); stop.append(a['kind']=='stop')
    def tensor(v,dtype): return torch.as_tensor(v,dtype=dtype,device=device)
    return dict(owners=tensor(owners,torch.long),pool_action=tensor(pool_action,torch.long),
                features=tensor(np.asarray(feats).reshape(-1,ACTION_WIDTH),torch.float32),batch=tensor(a_batch,torch.long),
                stop=tensor(stop,torch.bool),counts=counts)
