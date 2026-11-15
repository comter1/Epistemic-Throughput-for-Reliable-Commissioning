"""Content-addressed evidence, atomic writes, and strict prerequisite checks."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import time
import traceback
import fcntl

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def hash_json(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()

canonical_hash = hash_json

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

file_hash = file_sha256

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def read_json(path):
    return json.loads(Path(path).read_text())

def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(canonical_bytes(obj).decode() + "\n")
        handle.flush()
        os.fsync(handle.fileno())

def source_hash(experiment):
    paths = sorted((Path(experiment) / "src").rglob("*.py"))
    return hash_json({str(p.relative_to(experiment)): file_sha256(p) for p in paths})

def verify_hashes(hashes, base=None):
    for path, expected in hashes.items():
        p = Path(base, path) if base else Path(path)
        if not p.is_file() or file_sha256(p) != expected:
            raise ValueError(f"Missing or changed prerequisite: {p}")

def require_protocol_lock(root, config=None, include_test=False):
    root = Path(root)
    lock = read_json(root / "protocol_lock.json")
    if lock.get("state") != "passed":
        raise ValueError("Protocol lock is not qualified")
    if config is not None and hash_json(config) != lock["config_hash"]:
        raise ValueError("Configuration differs from frozen protocol")
    if not lock.get("technical_smoke", False):
        from .config import EXPERIMENT
        if lock.get("source_hash") != source_hash(EXPERIMENT):
            raise ValueError("Source code changed after sealing; a logged versioned amendment is required")
        current_source_manifest = root / "software/source_files.json"
        if not current_source_manifest.is_file():
            raise ValueError("Sealed source-file inventory is absent")
        verify_hashes(read_json(current_source_manifest), EXPERIMENT)
    verify_hashes(lock["files"], root)
    if include_test:
        verify_hashes(lock.get("sealed_private_test_files", {}), root)
    return lock

@contextmanager
def command_record(root, arguments, config):
    from .config import EXPERIMENT
    root = Path(root)
    identifier = f"{time.time_ns()}-{os.getpid()}"
    path = root / "commands" / f"{identifier}.json"
    started = time.monotonic()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXPERIMENT, text=True, capture_output=True)
    patch = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=EXPERIMENT, capture_output=True).stdout
    patch_path = root / "commands" / f"{identifier}.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(patch)
    record = {"command_arguments": arguments, "started_at": now(), "state": "running", "pid": os.getpid(),
              "configuration": config, "config_hash": hash_json(config), "source_hash": source_hash(EXPERIMENT),
              "git_commit": git.stdout.strip(), "dirty_patch": str(patch_path), "dirty_patch_hash": file_sha256(patch_path),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "outputs": {}}
    atomic_json(path, record)
    try:
        yield record
    except BaseException as exc:
        record.update(state="failed", exit_code=1, failure_reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    else:
        record.update(state="passed", exit_code=0)
    finally:
        record.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        atomic_json(path, record)
