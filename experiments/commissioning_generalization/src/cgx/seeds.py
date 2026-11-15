"""Domain-separated random streams independent of process hash randomization."""
import hashlib
import json

def stable_seed(*parts):
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(b"CGX-1.0\0" + raw).digest()[:8], "little") & ((1 << 63) - 1)

seed_for = stable_seed
