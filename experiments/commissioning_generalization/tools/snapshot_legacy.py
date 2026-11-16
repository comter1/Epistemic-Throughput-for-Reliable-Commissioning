#!/usr/bin/env python3
"""Hash protected legacy folders without importing or modifying them."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PROTECTED = ("finite_delay_benchmark_complete", "finite-delay-plotting", "result_review")

def snapshot(repo: Path) -> dict:
    repo = repo.resolve()
    result = {}
    for folder in PROTECTED:
        path = repo / folder
        if not path.is_dir():
            raise FileNotFoundError(f"Missing protected directory: {path}")
        for p in sorted(path.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            result[p.relative_to(repo).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"protected_folders": list(PROTECTED), "files": result}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path)
    args = ap.parse_args()
    current = snapshot(args.repo)
    if args.compare:
        old = json.loads(args.compare.read_text(encoding="utf-8"))["files"]
        new = current["files"]
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        current["comparison"] = {"unchanged": not changed, "changed_paths": changed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(current, indent=2)+"\n", encoding="utf-8")
    print(f"Hashed {len(current['files'])} protected files -> {args.out}")
    if args.compare and not current["comparison"]["unchanged"]:
        raise SystemExit("Protected files changed; inspect output before proceeding")

if __name__ == "__main__":
    main()
