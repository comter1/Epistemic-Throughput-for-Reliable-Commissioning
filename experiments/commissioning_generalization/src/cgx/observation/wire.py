"""Actual CGX v1 wire serialization, never simulator scheduling metadata."""
import struct
import numpy as np
EVENT = struct.Struct('<d16sIBBH14f40x')
BURST = struct.Struct('<16d4Q')
EVENT_CODES = {'receive': 1, 'send': 2, 'sample': 3}

def encode_record(record, port_index, channels=range(7)):
    values = list(record['values'])
    if len(values) > 14:
        raise ValueError('oversize public payload')
    values += [0.] * (14 - len(values))
    token = bytes.fromhex(record['message_token'])
    if len(token) != 16:
        raise ValueError('message token must be 16 opaque bytes')
    return EVENT.pack(record['local_time_s'], token, port_index, EVENT_CODES[record['event_type']],
                      int(record['missing']), sum(1 << int(c) for c in channels), *values)

def encode_burst(burst, domain_indices):
    values = np.asarray(burst['timestamps_s'], dtype=float)
    if values.shape != (4, 4):
        raise ValueError('calibration requires four four-timestamp exchanges')
    i, j = [domain_indices[d] for d in burst['pair']]
    return BURST.pack(*values.ravel(), i, j, 0, 0)

def offset_estimate(burst):
    t = np.asarray(burst['timestamps_s'], dtype=float)
    return ((t[:, 1] - t[:, 0]) + (t[:, 2] - t[:, 3])) / 2
