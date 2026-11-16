"""Public measurement identity, independent of numerical value or file layout.

Ingress assigns IDs from an actual acquisition identity and an event occurrence.
Once assigned, IDs must accompany records through all file/block transformations.
Opaque IDs are used for equality only and never become numerical model inputs.
"""
from collections import defaultdict
from copy import deepcopy
import math
from cgx.provenance import hash_json

EVENT_VERSION = 'cgx.structured.events.v1'


def _deduplicate(items, identity='measurement_event_id'):
    seen = {}
    for item in items:
        key = item[identity]
        if key in seen and seen[key] != item:
            raise ValueError(f'Conflicting values for the same measurement identity: {key}')
        seen[key] = item
    return [seen[k] for k in sorted(seen)]


def canonical_evidence(evidence):
    """Union immutable events. Repeated storage contributes exactly once."""
    if evidence['event_version'] != EVENT_VERSION:
        raise ValueError('Unknown structured event version')
    return dict(event_version=EVENT_VERSION, graph=evidence['graph'],
                events=_deduplicate(evidence.get('events', [])),
                calibrations=_deduplicate(evidence.get('calibrations', [])),
                statuses=_deduplicate(evidence.get('statuses', [])))


def prepare_history(history, provenance=None):
    """Annotate legacy public records using proven acquisition identities.

    ``provenance['observation_ids']`` aligns with the original observation blocks.
    Calibration-only legacy observations have no opaque tokens and therefore
    require provenance; hashing their numerical timestamps is prohibited.
    An execution-only fallback uses its public external random message tokens.
    It is intended for ordinary un-repackaged legacy observations, not for
    reconstructing identity after a lossy storage transformation.
    """
    graph = history['graph']
    ports = {n['id']+':'+p['name']: dict(component_id=n['id'],
             clock_group_id=n['clock_domain'], unit=p['unit'])
             for n in graph['nodes'] for p in n['ports']}
    ports['external:out'] = dict(component_id='external',
        clock_group_id=graph['clock_domains'][0], unit='position_m_reference_m_flags')
    provenance = provenance if provenance is not None else history.get('structured_provenance')
    observation_ids = None if provenance is None else provenance['observation_ids']
    observations = history.get('observations', [])
    if observation_ids is not None and len(observation_ids) != len(observations):
        raise ValueError('Observation provenance does not align with acquired blocks')
    events, calibrations, statuses = [], [], []
    for oi, obs in enumerate(observations):
        if observation_ids is not None:
            acquisition = str(observation_ids[oi])
        elif obs.get('execution_id'):
            acquisition = str(obs['execution_id'])
        else:
            tokens = [r['message_token'] for r in obs.get('records', [])
                      if r['port'] == 'external:out']
            if not tokens:
                if not obs.get('records') and not obs.get('calibration_bursts'):
                    # Choosing or stopping without a measurement changes no H.
                    continue
                raise ValueError('Legacy observation requires actual acquisition provenance')
            if obs.get('calibration_bursts'):
                raise ValueError('Calibration acquisition identity must be explicitly supplied')
            acquisition = hash_json([EVENT_VERSION, 'public_external_tokens', tokens])
        occurrences = defaultdict(int)
        for raw in obs.get('records', []):
            port = raw['port']
            if port not in ports:
                raise ValueError(f'Unknown public port: {port}')
            event_type = raw['event_type']
            occurrence = occurrences[port, event_type]
            occurrences[port, event_type] += 1
            event_id = raw.get('measurement_event_id') or hash_json(
                [EVENT_VERSION, acquisition, 'record', port, event_type, occurrence])
            values = [float(v) for v in raw['values']]
            timestamp = float(raw['local_time_s'])
            if not all(math.isfinite(v) for v in values+[timestamp]):
                raise ValueError('Nonfinite acquired physical value')
            events.append(dict(measurement_event_id=event_id, message_event_id=event_id,
                execution_id=str(raw.get('execution_id', acquisition)),
                message_token=raw['message_token'], port_id=port, **ports[port],
                observation_type=event_type, physical_value=values,
                timestamp=timestamp, missing=bool(raw['missing']), timeout=False))
        for bi, burst in enumerate(obs.get('calibration_bursts', [])):
            ids = burst.get('measurement_event_ids')
            for ei, timestamps in enumerate(burst['timestamps_s']):
                event_id = ids[ei] if ids else hash_json(
                    [EVENT_VERSION, acquisition, 'calibration', bi, ei])
                calibrations.append(dict(measurement_event_id=event_id,
                    execution_id=str(burst.get('execution_id', acquisition)),
                    pair=list(burst['pair']), timestamps_s=list(map(float,timestamps)),
                    unit='s', observation_type='four_timestamp_calibration',
                    missing=False, timeout=False))
        if obs.get('visible_status') in ('timeout', 'truncated'):
            statuses.append(dict(measurement_event_id=hash_json([EVENT_VERSION,acquisition,'status']),
                execution_id=acquisition, observation_type=obs['visible_status'],
                missing=True, timeout=obs['visible_status']=='timeout'))
    return canonical_evidence(dict(event_version=EVENT_VERSION, graph=graph,
        events=events, calibrations=calibrations, statuses=statuses))


def merge_evidence(blocks):
    """Repackage previously identified evidence without inventing acquisitions."""
    if not blocks:
        raise ValueError('At least one evidence block is required')
    graph = blocks[0]['graph']
    if any(b['graph'] != graph for b in blocks):
        raise ValueError('Cannot merge different public system graphs')
    return canonical_evidence(dict(event_version=EVENT_VERSION,graph=graph,
        events=[e for b in blocks for e in b.get('events',[])],
        calibrations=[e for b in blocks for e in b.get('calibrations',[])],
        statuses=[e for b in blocks for e in b.get('statuses',[])]))


def annotate_history(history,provenance):
    """Persist ingress IDs in raw records BEFORE a split/merge storage test.

    The returned object is an adapter-side representation, never written into
    the unchanged legacy Store schema. Re-reading it with prepare_history
    preserves event IDs even after records or exchanges change block positions.
    """
    result=deepcopy(history)
    ids=provenance['observation_ids']
    if len(ids)!=len(result['observations']):raise ValueError('Provenance alignment')
    for acquisition,obs in zip(ids,result['observations']):
        acquisition=str(acquisition);obs['execution_id']=acquisition
        occurrence=defaultdict(int)
        for raw in obs.get('records',[]):
            key=raw['port'],raw['event_type'];ordinal=occurrence[key];occurrence[key]+=1
            raw.setdefault('execution_id',acquisition)
            raw.setdefault('measurement_event_id',hash_json([EVENT_VERSION,acquisition,'record',*key,ordinal]))
        for bi,burst in enumerate(obs.get('calibration_bursts',[])):
            burst.setdefault('execution_id',acquisition)
            burst.setdefault('measurement_event_ids',[hash_json([EVENT_VERSION,acquisition,'calibration',bi,ei])
                for ei in range(len(burst['timestamps_s']))])
    result.pop('structured_provenance',None)
    return result


def evidence_identity(evidence):
    """Cache identity includes event identities and values; never used as a feature."""
    return hash_json(canonical_evidence(evidence))


def duplicate_storage(evidence, copies=1):
    result = deepcopy(evidence)
    for key in ('events','calibrations','statuses'):
        result[key] = result.get(key, []) * (copies+1)
    return result
