"""Learned local likelihoods with exact linear-Gaussian nuisance integration.

Timing model, within an opaque packet identity and token-preserving graph segment:
    t_e = U_packet + C_clock(e) + D_fault * downstream(e, fault) + epsilon_e.
The packet epoch U is eliminated with a flat prior by weighted centering. Clock
offsets C persist across every acquisition. D also persists, with a learned small
Gaussian-mixture prior. This retains the covariance between timestamp contrasts;
it never assigns an independent clock bias to each measurement or each block.
"""
from collections import defaultdict
from functools import lru_cache
import math
import numpy as np
import networkx as nx
from scipy.special import logsumexp
from cgx.models.features import target_space
from cgx.provenance import hash_json
from .events import canonical_evidence

LIKELIHOOD_VERSION = 'cgx.structured.linear_gaussian.v1'


def topology(graph):
    """Public graph determines packet segments, candidate paths and signal units."""
    nodes = {n['id']:n for n in graph['nodes']}
    ports = {n['id']+':'+p['name']:p for n in graph['nodes'] for p in n['ports']}
    dag = nx.DiGraph(); dag.add_nodes_from(nodes)
    pd = nx.DiGraph(); pd.add_nodes_from(ports)
    incoming = defaultdict(list)
    for e in graph['edges']:
        dag.add_edge(e['source'],e['target']); incoming[e['target']].append(e)
        pd.add_edge(e['source']+':'+e['source_port'],e['target']+':'+e['target_port'])
    for n in graph['nodes']:
        # Controllers and mergers issue a NEW opaque packet token.
        if n['type'] in ('controller','merger'):
            continue
        for a in n['ports']:
            for b in n['ports']:
                if a['direction']=='in' and b['direction']=='out' and set(a['channels'])&set(b['channels']):
                    pd.add_edge(n['id']+':'+a['name'],n['id']+':'+b['name'])
    segment = {p:min(component) for component in nx.weakly_connected_components(pd) for p in component}
    affines = {}
    for nid in nx.lexicographical_topological_sort(dag):
        n=nodes[nid]
        for e in incoming[nid]:
            affines[nid+':'+e['target_port']]=affines[e['source']+':'+e['source_port']]
        if n['type']=='adapter':
            scale=float(n['nominal'].get('scale',1));offset=float(n['nominal'].get('offset',0))
        elif n['type'] in ('sensor','controller','merger'):
            scale,offset=1.,0.
        else:
            inputs=[affines[nid+':'+p['name']] for p in n['ports'] if p['direction']=='in']
            scale,offset=inputs[0] if inputs else (1.,0.)
        for p in n['ports']:
            if p['direction']=='out':affines[nid+':'+p['name']]=(scale,offset)
    targets,_,_,prior=target_space(graph)
    delay_nodes=[key.split(':')[0] for key in targets if key.endswith((':cache_stale',':queue_hold'))]
    descendants={}
    for nid in delay_nodes:
        reach=set()
        for p in nodes[nid]['ports']:
            if p['direction']=='out':
                port=nid+':'+p['name'];reach.add(port);reach.update(nx.descendants(pd,port))
        descendants[nid]=reach
    return dict(nodes=nodes,ports=ports,segment=segment,affines=affines,
        clocks=list(graph['clock_domains']),targets=targets,prior=np.asarray(prior),
        delay_nodes=delay_nodes,descendants=descendants)


def _values(record,spec):
    channels=list(spec['channels'])
    if spec['unit']=='state_rad':channels+= [c+7 for c in channels]
    values=record['physical_value']
    if len(values)!=len(channels):raise ValueError('Signal payload contradicts public port channels/unit')
    return dict(zip(channels,values))


def prepare_evidence(evidence):
    """Model-independent sufficient public representation; no labels or actions."""
    e=canonical_evidence(evidence); meta=topology(e['graph'])
    clock_lookup={c:i for i,c in enumerate(meta['clocks'])}
    records=[r for r in e['events'] if not r['missing'] and not r['timeout'] and r['port_id']!='external:out']
    group_keys=sorted({(r['execution_id'],r['message_token'],meta['segment'][r['port_id']]) for r in records})
    groups={g:i for i,g in enumerate(group_keys)}
    tt=np.asarray([r['timestamp'] for r in records],dtype=np.float64)
    cc=np.asarray([clock_lookup[r['clock_group_id']] for r in records],dtype=np.int64)
    gg=np.asarray([groups[r['execution_id'],r['message_token'],meta['segment'][r['port_id']]] for r in records],dtype=np.int64)
    path=np.asarray([[r['port_id'] in meta['descendants'][nid] for nid in meta['delay_nodes']] for r in records],dtype=np.float64)
    if not records:path=np.empty((0,len(meta['delay_nodes'])),dtype=np.float64)
    local=defaultdict(lambda:defaultdict(list))
    for r in records:
        n=meta['nodes'][r['component_id']]
        if n['type'] in ('adapter','cache','queue'):
            local[r['execution_id'],r['message_token'],r['component_id']][r['port_id']].append(r)
    gain=defaultdict(list); delay=defaultdict(list); gain_group_ids=defaultdict(list)
    for key,by_port in sorted(local.items()):
        nid=key[2];node=meta['nodes'][nid]
        ins=[p for p in by_port if meta['ports'][p]['direction']=='in']
        outs=[p for p in by_port if meta['ports'][p]['direction']=='out']
        # A local module has one input and one output in the preserved benchmark.
        # Multiple physical readings of the same packet share its payload; this
        # local signal factor is conditional on that shared packet, not one
        # independent draw of the physical signal for each storage appearance.
        for inp in sorted(ins):
            for outp in sorted(outs):
                if node['type'] in ('cache','queue'):
                    delay[nid].append(float(np.mean([r['timestamp'] for r in by_port[outp]])-
                        np.mean([r['timestamp'] for r in by_port[inp]])))
                    continue
                iv=[_values(r,meta['ports'][inp]) for r in by_port[inp]]
                ov=[_values(r,meta['ports'][outp]) for r in by_port[outp]]
                channels=sorted(set(iv[0])&set(ov[0]))
                if not channels:continue
                x=np.array([np.mean([v[c] for v in iv]) for c in channels])
                y=np.array([np.mean([v[c] for v in ov]) for c in channels])
                scale_in,offset_in=meta['affines'][inp]
                expected=float(node['nominal'].get('scale',1.))*(x-offset_in)/scale_in+float(node['nominal'].get('offset',0.))
                norm=float(expected@expected)
                if norm<=np.finfo(float).tiny:continue
                gain[nid].append(float(expected@(y-expected)/norm))
                gain_group_ids[nid].append(hash_json([key,inp,outp]))
    calibrations=[]
    for b in e['calibrations']:
        a,z=b['pair'];t=np.asarray(b['timestamps_s'],dtype=np.float64)
        if a not in clock_lookup or z not in clock_lookup:raise ValueError('Unknown public calibration clock')
        calibrations.append((clock_lookup[a],clock_lookup[z],float(((t[1]-t[0])+(t[2]-t[3]))/2)))
    return dict(prepared_version=LIKELIHOOD_VERSION,meta=meta,times=tt,clocks=cc,
        groups=gg,group_count=len(groups),delay_path=path,
        gain={n:np.asarray(v,dtype=np.float64) for n,v in gain.items()},
        local_delay={n:np.asarray(v,dtype=np.float64) for n,v in delay.items()},
        calibration=np.asarray(calibrations,dtype=np.float64).reshape(-1,3),
        unique_events=len(e['events']),unique_calibrations=len(e['calibrations']),
        missing_events=sum(r['missing'] or r['timeout'] for r in e['events'])+len(e['statuses']),
        external_events=sum(r['port_id']=='external:out' for r in e['events']),
        signal_packet_groups=sum(len(x) for x in gain.values()),
        gain_group_ids=dict(gain_group_ids))


def combine_prepared_acquisitions(blocks):
    """Join disjoint, complete acquisitions BEFORE any nuisance integration.

    The caller establishes disjoint physical acquisition identities, and never
    sends split event packages here. Message epochs are distinct across these
    acquisitions; clock and fault latents remain common in ``posterior``. This
    is an exact preparation cache, not a product of independently marginalized
    block likelihoods.
    """
    if not blocks:raise ValueError('At least one prepared acquisition is required')
    first=blocks[0]
    result=dict(prepared_version=LIKELIHOOD_VERSION,meta=first['meta'])
    for key in ('times','clocks','delay_path','calibration'):
        result[key]=np.concatenate([b[key] for b in blocks],axis=0)
    offset=0;groups=[]
    for b in blocks:
        groups.append(b['groups']+offset);offset+=b['group_count']
    result['groups']=np.concatenate(groups);result['group_count']=offset
    for key in ('gain','local_delay'):
        nodes=sorted({n for b in blocks for n in b[key]})
        result[key]={n:np.concatenate([b[key][n] for b in blocks if n in b[key]]) for n in nodes}
    nodes=sorted({n for b in blocks for n in b['gain_group_ids']})
    result['gain_group_ids']={n:[g for b in blocks for g in b['gain_group_ids'].get(n,[])] for n in nodes}
    for key in ('unique_events','unique_calibrations','missing_events','external_events','signal_packet_groups'):
        result[key]=sum(b[key] for b in blocks)
    return result


def timing_normal_equations(prepared,parameters,fault_node=None,fault_family=None):
    """Eliminate every message epoch with its exact correlated contrast metric."""
    p=prepared;k=len(p['meta']['clocks']);n=len(p['times']);ng=p['group_count']
    if fault_node is not None:
        index=p['meta']['delay_nodes'].index(fault_node);v=p['delay_path'][:,index]
        extra=float(parameters['timing_process_variance'].get(fault_family,0.))
        width=k+1
    else:v=np.zeros(n);extra=0.;width=k
    variance=float(parameters['timestamp_noise_variance'])+extra*v
    S=np.zeros((width,width));b=np.zeros(width);q=0.;constant=0.
    if n:
        w=1./variance;g=p['groups'];c=p['clocks'];t=p['times']
        # Center timestamps before forming y'y; no near-cancelling 10-second
        # epochs enter the small residual sums or clock precision equations.
        arithmetic=np.bincount(g,weights=t,minlength=ng)/np.bincount(g,minlength=ng)
        y=t-arithmetic[g]
        sumw=np.bincount(g,weights=w,minlength=ng)
        sumwy=np.bincount(g,weights=w*y,minlength=ng)
        gx=np.zeros((ng,width))
        gx[:,:k]=np.bincount(g*k+c,weights=w,minlength=ng*k).reshape(ng,k)
        S[np.arange(k),np.arange(k)]=np.bincount(c,weights=w,minlength=k)
        b[:k]=np.bincount(c,weights=w*y,minlength=k)
        if fault_node is not None:
            gx[:,k]=np.bincount(g,weights=w*v,minlength=ng)
            cross=np.bincount(c,weights=w*v,minlength=k)
            S[:k,k]=cross;S[k,:k]=cross;S[k,k]=np.sum(w*v*v)
            b[k]=np.sum(w*v*y)
        S-=gx.T@(gx/sumw[:,None]);b-=gx.T@(sumwy/sumw)
        q=float(np.sum(w*y*y)-np.sum(sumwy*sumwy/sumw))
        constant=-.5*((n-ng)*math.log(2*math.pi)+np.log(variance).sum()+np.log(sumw).sum())
    calibration_variance=float(parameters['calibration_noise_variance'])
    for a,z,y in p['calibration']:
        x=np.zeros(width);x[int(a)]=-1.;x[int(z)]=1.
        S+=np.outer(x,x)/calibration_variance;b+=x*y/calibration_variance;q+=y*y/calibration_variance
        constant-=.5*math.log(2*math.pi*calibration_variance)
    return S,b,q,float(constant)


def integrate_gaussian(S,b,q,constant,means,variances):
    """Exact Gaussian marginal log likelihood and latent posterior moments."""
    means=np.asarray(means,dtype=np.float64);variances=np.asarray(variances,dtype=np.float64)
    precision=S+np.diag(1./variances)
    rhs=b+means/variances
    sign,logdet=np.linalg.slogdet(precision)
    if sign<=0:raise FloatingPointError('Nonpositive latent precision')
    covariance=np.linalg.inv(precision);mean=covariance@rhs
    quadratic=float(q+np.sum(means*means/variances)-rhs@mean)
    ll=constant-.5*(np.log(variances).sum()+logdet+quadratic)
    return float(ll),mean,covariance


def integrated_gain(values,parameters,mixture=None):
    """Common gain parameter, shared across new messages and acquisitions."""
    y=np.asarray(values,dtype=np.float64);variance=float(parameters['gain_noise_variance'])
    if not len(y):return 0.
    center=float(parameters['healthy_gain_mean'])
    z=y-center
    base=-.5*(len(y)*math.log(2*math.pi*variance)+float(z@z)/variance)
    if mixture is None:return float(base)
    # Stable Bayes factor against the identical healthy likelihood. A shared
    # latent gain, not a fresh fault severity per observation, is integrated.
    s=len(y)/variance;b=float(z.sum())/variance
    terms=[]
    for weight,mu,var in zip(mixture['weights'],mixture['means'],mixture['variances']):
        precision=s+1./var;rhs=b+(mu-center)/var
        logratio=-.5*(math.log(var)+math.log(precision)+(mu-center)**2/var-rhs*rhs/precision)
        terms.append(math.log(weight)+logratio)
    return float(base+logsumexp(terms))


def posterior(prepared,parameters,include_latents=True):
    meta=prepared['meta'];k=len(meta['clocks']);prior_clock=float(parameters['clock_prior_variance'])
    S,b,q,c=timing_normal_equations(prepared,parameters)
    base_t,base_mean,base_cov=integrate_gaussian(S,b,q,c,np.zeros(k),np.full(k,prior_clock))
    base_gain={nid:integrated_gain(v,parameters) for nid,v in prepared['gain'].items()}
    logs=[];latent=[];component_details=[]
    for target in meta['targets']:
        extra_gain=0.
        if target=='healthy' or target.endswith(':adapter_scale'):
            value=base_t;cm,cv=base_mean,base_cov
            if target!='healthy':
                nid=target.split(':')[0];values=prepared['gain'].get(nid,np.empty(0))
                extra_gain=integrated_gain(values,parameters,parameters['gain_mixture'])-base_gain.get(nid,0.)
            details=[]
        else:
            nid,family=target.split(':');S,b,q,c=timing_normal_equations(prepared,parameters,nid,family)
            mixture=parameters['delay_mixtures'][family];parts=[];means=[];covs=[]
            for weight,mu,var in zip(mixture['weights'],mixture['means'],mixture['variances']):
                ll,mean,cov=integrate_gaussian(S,b,q,c,np.r_[np.zeros(k),mu],np.r_[np.full(k,prior_clock),var])
                parts.append(math.log(weight)+ll);means.append(mean);covs.append(cov)
            value=float(logsumexp(parts));weights=np.exp(np.asarray(parts)-value)
            cm=sum(w*m[:k] for w,m in zip(weights,means))
            cv=sum(w*(v[:k,:k]+np.outer(m[:k]-cm,m[:k]-cm)) for w,m,v in zip(weights,means,covs))
            details=[dict(weight=float(w),delay_mean=float(m[-1]),delay_variance=float(v[-1,-1])) for w,m,v in zip(weights,means,covs)]
        logs.append(value-base_t+extra_gain);latent.append((cm,cv));component_details.append(details)
    logposterior=np.log(meta['prior'])+np.asarray(logs);logposterior-=logsumexp(logposterior)
    probs=np.exp(logposterior);floor=float(parameters.get('probability_floor',1e-8))
    probs=np.maximum(probs,floor);probs/=probs.sum()
    result=dict(target_keys=list(meta['targets']),log_probabilities_base2=np.log2(probs).tolist())
    if include_latents:
        clock_mean=sum(p*m for p,(m,v) in zip(probs,latent))
        clock_cov=sum(p*(v+np.outer(m-clock_mean,m-clock_mean)) for p,(m,v) in zip(probs,latent))
        result['structured_diagnostics']=dict(clock_group_ids=meta['clocks'],
            clock_prior_variance=prior_clock,clock_mean=clock_mean.tolist(),clock_covariance=clock_cov.tolist(),
            unique_measurement_events=prepared['unique_events'],unique_calibration_events=prepared['unique_calibrations'],
            timing_contrasts=len(prepared['times'])-prepared['group_count'],
            signal_packet_groups=prepared['signal_packet_groups'],missing_events=prepared['missing_events'],
            external_events=prepared['external_events'],relative_log_likelihoods=logs,
            delay_component_posteriors=component_details,
            timestamp_noise_variance=parameters['timestamp_noise_variance'],
            calibration_noise_variance=parameters['calibration_noise_variance'])
    return result


def fit_mixture(values,components,variance_floor,iterations=100):
    """Deterministic one-dimensional Gaussian-mixture maximum likelihood."""
    x=np.asarray(values,dtype=np.float64)
    if len(x)<components:raise ValueError('Insufficient source cases for declared mixture')
    means=np.quantile(x,(np.arange(components)+.5)/components)
    variances=np.full(components,max(float(np.var(x))/components,variance_floor))
    weights=np.full(components,1./components)
    for _ in range(iterations):
        lp=np.log(weights)[None,:]-.5*(np.log(2*np.pi*variances)[None,:]+(x[:,None]-means[None,:])**2/variances[None,:])
        responsibilities=np.exp(lp-logsumexp(lp,axis=1)[:,None]);mass=responsibilities.sum(0)+1e-12
        weights=mass/mass.sum();means=(responsibilities*x[:,None]).sum(0)/mass
        variances=np.maximum((responsibilities*(x[:,None]-means)**2).sum(0)/mass,variance_floor)
    order=np.argsort(means)
    return dict(weights=weights[order].tolist(),means=means[order].tolist(),variances=variances[order].tolist(),
        fitted_cases=len(x),iterations=iterations,variance_floor=float(variance_floor))
