#!/usr/bin/env python3
"""Build executable jobs only from sealed, actually prepared V2 prerequisites."""
import argparse
from pathlib import Path
from cgx.config import load_config,DEFAULT_CONFIG
from cgx.runtime.queue import build_jobs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['readers','policies'],default='policies')
    a=p.parse_args();print(build_jobs(a.data_root,load_config(a.config),a.phase,a.out))
