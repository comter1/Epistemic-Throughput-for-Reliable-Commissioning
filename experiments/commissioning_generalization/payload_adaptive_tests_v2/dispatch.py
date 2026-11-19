"""Light local SSH coordinator; heavy work runs only on currently authorized nodes."""
import json
from pathlib import Path
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from .protocol import EXP, write_json, digest

PYTHON='/home/leiyo/work/cgx_env/bin/python'

def dispatch(root,jobs,workers=(6,2),tag='jobs'):
    root=Path(root); root.joinpath('commands').mkdir(exist_ok=True)
    total=sum(workers); cut=round(len(jobs)*workers[0]/total)
    chunks=[jobs[:cut],jobs[cut:]]
    def node(port,count,chunk):
        if not chunk: return dict(port=port,exit_code=0,jobs=0)
        path=root/'commands'/f'{tag}_{port}.json'; write_json(path,chunk)
        remote='cd '+shlex.quote(str(EXP))+' && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= '+PYTHON+' -m payload_adaptive_tests_v2.worker --root '+shlex.quote(str(root))+' --jobs '+shlex.quote(str(path))+' --workers '+str(count)
        command=['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','-p',str(port),'leiyo@ssh.cloud.sdu.dk',remote]
        write_json(root/'commands'/f'{tag}_{port}_command.json',dict(command=command,job_count=len(chunk),workers=count))
        with (root/'commands'/f'{tag}_{port}.log').open('w') as out:
            result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
        return dict(port=port,exit_code=result.returncode,jobs=len(chunk))
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda x:node(*x),zip([2072,2379],workers,chunks)))
    write_json(root/'commands'/f'{tag}_exit_codes.json',results)
    if any(r['exit_code'] for r in results): raise RuntimeError('Remote stage failed: '+str(results))
    return results
