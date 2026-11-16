"""Explicit configuration and artifact locations."""
from pathlib import Path
import json
import os

EXPERIMENT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = EXPERIMENT / "configs/protocol.json"

def load_config(path=None):
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    if config["protocol_id"] not in ("CGX-1.0","CGX-2.0"):
        raise ValueError("Unsupported protocol; implement a versioned amendment explicitly")
    return config

def data_root(path=None):
    value = path or os.environ.get("CGX_DATA_ROOT")
    if not value:
        raise ValueError("Set CGX_DATA_ROOT or --data-root to an external artifact directory")
    root = Path(value).expanduser().resolve()
    repository = EXPERIMENT.parents[1]
    if root == repository or repository in root.parents:
        raise ValueError("Generated artifacts must live outside the repository, including protected legacy directories")
    root.mkdir(parents=True, exist_ok=True)
    return root

def allocated_cpu_count():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return count

def default_workers():
    return min(16, max(1, allocated_cpu_count() // 2))
