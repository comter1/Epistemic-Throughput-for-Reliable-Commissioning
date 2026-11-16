#!/usr/bin/env python3
"""Read-only host probe. Does not install software, benchmark GPUs, or expose secrets."""
from __future__ import annotations
import argparse
import importlib.metadata as im
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

def run(command: list[str]) -> dict:
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"command": command, "exit_code": r.returncode,
                "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": command, "error": str(e)}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--disk-path", type=Path, default=Path.cwd())
    args = ap.parse_args()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    d = shutil.disk_usage(args.disk_path)
    packages = {}
    for name in ("torch", "mujoco", "numpy", "scipy", "networkx", "pyarrow", "pytest"):
        try:
            packages[name] = im.version(name)
        except im.PackageNotFoundError:
            packages[name] = None
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": platform.platform(), "cpu_count_advertised": os.cpu_count(),
              "cpu_affinity": affinity, "disk_path": str(args.disk_path.resolve()),
              "disk_free_gib": d.free / 2**30, "packages": packages,
              "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,utilization.gpu", "--format=csv,noheader"])}
    for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max"):
        try:
            report[f] = Path(f).read_text().strip()
        except OSError:
            pass
    try:
        report["memory_total_kib"] = int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')).split()[1])
    except (OSError, StopIteration, ValueError):
        pass
    if packages.get("torch"):
        report["torch_device_probe"] = run([sys.executable, "-c",
            "import torch,json; print(json.dumps({'version':torch.__version__,'cuda_build':torch.version.cuda,'available':torch.cuda.is_available(),'count':torch.cuda.device_count(),'devices':[{'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]}))"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(args.out)

if __name__ == "__main__":
    main()
