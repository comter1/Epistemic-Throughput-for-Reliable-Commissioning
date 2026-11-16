"""Observer-only clocks; independent timestamp errors and transport asymmetry."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], 'little')

def calibration_burst(pair, offsets_ms, seed, burst_index=0, *, noiseless=False, symmetric=False):
    rng = np.random.default_rng(stable_seed(seed, 'calibration', burst_index, sorted(pair)))
    variance = math.log(1 + (.5 / 2.) ** 2)
    mu = math.log(.002) - variance / 2
    sigma = math.sqrt(variance)
    rows = []
    # A hidden random local epoch avoids a known simulator-start marker.
    send = float(rng.uniform(1, 20))
    start = send
    for _ in range(4):
        forward = float(rng.lognormal(mu, sigma))
        backward = forward if symmetric else float(rng.lognormal(mu, sigma))
        private = np.array([send, send + forward, send + forward + .0001,
                            send + forward + .0001 + backward])
        offset = np.array([offsets_ms[pair[0]], offsets_ms[pair[1]], offsets_ms[pair[1]], offsets_ms[pair[0]]]) / 1000
        error = np.zeros(4) if noiseless else rng.normal(0, .001, 4)
        rows.append((private + offset + error).tolist())
        send = private[-1]
    return {'pair': list(pair), 'timestamps_s': rows}, send - start
