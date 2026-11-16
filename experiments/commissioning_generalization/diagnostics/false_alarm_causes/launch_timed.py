"""Run one diagnostic command and retain exact exit and child CPU accounting."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


if __name__=='__main__':
    destination=Path(sys.argv[1]);command=sys.argv[2:];started=time.monotonic()
    result=subprocess.run(command)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    destination.write_text(json.dumps({'command':command,'wall_seconds':time.monotonic()-started,
        'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,
        'max_rss_kb':usage.ru_maxrss,'exit_status':result.returncode},indent=2)+'\n')
    raise SystemExit(result.returncode)
