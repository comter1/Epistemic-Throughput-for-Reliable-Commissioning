"""Record the actual remote software and GPU/CPU allocation without secrets."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from cgx.provenance import atomic_json


def capture(run_dir, node):
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                if d.metadata.get('Name')}
    gpu = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version',
        '--format=csv'], text=True)
    result = dict(captured_at=datetime.now(timezone.utc).isoformat(), node=str(node),
        hostname=platform.node(), python=sys.version, executable=sys.executable,
        platform=platform.platform(), packages=packages, gpus=gpu,
        affinity_cores=len(os.sched_getaffinity(0)),
        cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        numerical_threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    path=Path(run_dir)/'provenance'/f'environment_{node}.json'
    if path.exists():
        raise ValueError('Original environment snapshot is preserved')
    atomic_json(path,result)
    print(node,platform.node(),result['cpu_max'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--node',required=True)
    a=p.parse_args();capture(a.run_dir,a.node)
