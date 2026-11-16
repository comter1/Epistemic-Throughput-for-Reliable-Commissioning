"""Environment snapshot and actual B200 BF16/scatter checks."""
from pathlib import Path
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from cgx.config import EXPERIMENT, allocated_cpu_count, default_workers
from cgx.provenance import atomic_json, file_sha256, now

def doctor(root, config):
    import torch
    import psutil
    root = Path(root)
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "mujoco", "numpy", "scipy", "networkx", "pandas", "pyarrow", "matplotlib", "pytest", "jsonschema", "psutil", "zstandard")}
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], text=True)
    lock_path = EXPERIMENT / "requirements.lock.txt"
    lock_path.write_text(frozen)
    asset = root / "assets/mujoco_menagerie"
    asset_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=asset, text=True).strip()
    panda = asset / "franka_emika_panda"
    assets = {str(p.relative_to(root)): file_sha256(p) for p in sorted(panda.rglob("*")) if p.is_file()}
    if not assets:
        raise RuntimeError("Panda asset absent")
    devices = []
    for index in range(torch.cuda.device_count()):
        started = time.monotonic()
        device = torch.device(f"cuda:{index}")
        torch.manual_seed(671918)
        x = torch.randn(128, 128, device=device, requires_grad=True)
        layer = torch.nn.Linear(128, 128).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = layer(x)
            ids = torch.arange(128, device=device) % 16
            summed = torch.zeros(16, 128, device=device, dtype=values.dtype).index_add(0, ids, values)
            loss = summed.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        passed = bool(torch.isfinite(loss) and torch.isfinite(x.grad).all())
        devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "bf16_forward_backward_scatter": "passed" if passed else "failed",
                        "warmup_seconds": time.monotonic() - started})
        if not passed:
            raise RuntimeError("BF16/scatter compatibility check failed")
    from .remote import registered_workers
    remotes=registered_workers()
    if len(devices)+len(remotes) != config["compute"]["gpus"]:
        raise RuntimeError("Doctor requires every registered local/remote GPU qualified")
    report = {"state": "passed", "time_utc": now(), "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": packages, "torch_cuda_build": torch.version.cuda,
              "devices": devices, "remote_devices":[{k:v for k,v in r.items() if k!='identity_file'} for r in remotes],"total_qualified_gpus":len(devices)+len(remotes), "cpu_affinity": sorted(os.sched_getaffinity(0)), "allocated_cpus": allocated_cpu_count(),
              "default_workers": default_workers(), "host_memory_bytes": psutil.virtual_memory().total,
              "disk_free_bytes": shutil.disk_usage(root).free, "asset_git_sha": asset_sha, "asset_hashes": assets,
              "dependency_lock_hash": file_sha256(lock_path),
              "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv"], text=True),
              "cgroups": {p: Path(p).read_text().strip() for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max") if Path(p).exists()}}
    atomic_json(root / "environment.json", report)
    atomic_json(EXPERIMENT / "reports/environment.json", report)
    return report
