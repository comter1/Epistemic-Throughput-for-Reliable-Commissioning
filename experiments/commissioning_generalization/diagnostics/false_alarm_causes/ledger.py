"""Task-local, cross-process accounting; reserve before an attempted operation."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import json
import os
import socket
import time


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    LIMITS = {"robot_execution": 18000, "model_and_features": 120000}
    KINDS = ("robot_execution", "calibration", "model_forward", "feature_build")

    def __init__(self, run):
        self.run = Path(run)
        self.run.mkdir(parents=True, exist_ok=True)
        self.path = self.run / "resource_ledger.json"
        self.lock = self.run / "resource_ledger.lock"

    def reserve(self, kind, count=1, detail=""):
        if kind not in self.KINDS or type(count) is not int or count < 0:
            raise ValueError("Invalid resource reservation")
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                "counts": {k: 0 for k in self.KINDS}, "limits": self.LIMITS,
                "by_detail": {}, "attempts_include_failures": True}
            proposed = dict(state["counts"])
            proposed[kind] += count
            if proposed["robot_execution"] > self.LIMITS["robot_execution"]:
                raise BudgetExceeded("18,000 new physical execution attempt limit reached")
            if proposed["model_forward"] + proposed["feature_build"] > self.LIMITS["model_and_features"]:
                raise BudgetExceeded("120,000 combined forward/feature-build attempt limit reached")
            state["counts"] = proposed
            label = str(detail).split(":", 1)[0] or "unspecified"
            stage = state["by_detail"].setdefault(label, {k: 0 for k in self.KINDS})
            stage[kind] += count
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            state["last_host"] = socket.gethostname()
            temporary = self.path.with_name(f".resource_ledger.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            temporary.replace(self.path)
            return dict(proposed)

    def snapshot(self):
        with self.lock.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            return json.loads(self.path.read_text()) if self.path.exists() else {
                "counts": {k: 0 for k in self.KINDS}, "limits": self.LIMITS}


class TimedStage:
    """Measured process CPU and wall time, separate from GPU allocation time."""
    def __init__(self, run, name):
        self.run, self.name = Path(run), name

    def __enter__(self):
        self.wall, self.cpu = time.monotonic(), time.process_time()
        return self

    def __exit__(self, kind, value, traceback):
        row = {"stage": self.name, "hostname": socket.gethostname(), "pid": os.getpid(),
               "wall_seconds": time.monotonic()-self.wall,
               "process_cpu_seconds": time.process_time()-self.cpu,
               "state": "passed" if kind is None else "failed",
               "error": None if kind is None else f"{kind.__name__}: {value}"}
        with (self.run / "stage_timings.jsonl").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(row) + "\n")
