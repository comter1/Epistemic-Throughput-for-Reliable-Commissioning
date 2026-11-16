"""CLI-facing qualification and auditable capacity-profile entry points."""
from pathlib import Path
from .qualification import qualify_system as _qualify, profile_executions
from .verification import public_verification_pass
from ..provenance import atomic_json, hash_json, now

def qualify_system(config,root,report=None):
    result=_qualify(config,task_count=32)
    result.update(state=result['status'],configuration_hash=hash_json(config),finished_at=now())
    path=Path(report) if report else Path(root)/'reports/qualification.json'
    atomic_json(path,result)
    if result['status']!='passed':
        raise RuntimeError(f'Healthy source qualification failed: {path}')
    return result
