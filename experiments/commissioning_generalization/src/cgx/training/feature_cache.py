"""Lossless memory-mapped feature cache: no pickle, quantization, or changed evidence."""
from pathlib import Path
import json
import os
import struct
import numpy as np
from cgx.models.features import GraphFeatures

MAGIC=b'CGXF0001'
META=('sequence_owners','targets','target_nodes','target_families','prior','physical_count','node_lookup','clock_lookup')


def write_features(path,features):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={key:np.ascontiguousarray(getattr(features,key)) for key in ('nodes','edge_index','edge_types','edge_features','resources','task')}
    arrays['sequences']=np.concatenate(features.sequences) if features.sequences else np.empty((0,64),np.float32)
    arrays['lengths']=np.asarray([len(s) for s in features.sequences],dtype=np.int64)
    specs={};offset=0
    for key,value in arrays.items():
        specs[key]=dict(shape=value.shape,dtype=value.dtype.str,offset=offset,nbytes=value.nbytes);offset+=value.nbytes
    header=json.dumps({'arrays':specs,'metadata':{key:getattr(features,key) for key in META}},sort_keys=True,separators=(',',':')).encode()
    padding=(-(16+len(header)))%64
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        stream.write(MAGIC+struct.pack('<Q',len(header))+header+b'\0'*padding)
        for value in arrays.values():stream.write(value.tobytes())
    os.replace(temp,path)


def read_features(path):
    path=Path(path)
    with path.open('rb') as stream:
        if stream.read(8)!=MAGIC:raise ValueError('Unknown feature cache format')
        length=struct.unpack('<Q',stream.read(8))[0]
        if length>1024*1024:raise ValueError('Invalid feature cache header')
        header=json.loads(stream.read(length))
    start=16+length+(-(16+length))%64;mapping=np.memmap(path,dtype=np.uint8,mode='r')
    arrays={}
    for key,spec in header['arrays'].items():
        dtype=np.dtype(spec['dtype'])
        if dtype.hasobject:raise ValueError('Object feature arrays forbidden')
        if int(np.prod(spec['shape']))*dtype.itemsize!=spec['nbytes'] or start+spec['offset']+spec['nbytes']>len(mapping):raise ValueError('Truncated feature cache')
        arrays[key]=np.ndarray(spec['shape'],dtype=dtype,buffer=mapping,offset=start+spec['offset'])
    data=arrays.pop('sequences');lengths=arrays.pop('lengths')
    sequences=np.split(data,np.cumsum(lengths)[:-1]) if len(lengths) else []
    return GraphFeatures(**arrays,sequences=sequences,**header['metadata'])
