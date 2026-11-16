"""Executed packet operators. Implementations A and B use independent delay machinery."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import numpy as np

@dataclass
class Packet:
    values: np.ndarray
    token: bytes
    channels: tuple[int, ...]
    state: bool
    scale: float = 1.
    offset: float = 0.
    missing: bool = False

    def subset(self, channels):
        return Packet(self.values.copy(), self.token, tuple(channels), self.state, self.scale, self.offset, self.missing)

    def canonical(self):
        return (self.values - self.offset) / self.scale

    def payload(self):
        channels = list(self.channels)
        idx = channels + [c + 7 for c in channels] if self.state else channels
        return self.values[idx].tolist()

class Module:
    def __init__(self, node, implementation='A'):
        self.node = node
        self.implementation = implementation
        self.delay_ticks = 0
        self.extra_gain = 1.
        self.reset()

    def reset(self):
        self.ring = deque(maxlen=64)
        self.snapshots = {}
        self.heap = []
        self.buckets = {}
        self.previous = None
        self.first = None

    def repair(self):
        self.delay_ticks = 0
        self.extra_gain = 1.
        self.reset()

    def cache(self, packet, tick):
        if self.implementation == 'A':
            self.ring.append(packet)
            return self.ring[max(0, len(self.ring) - 1 - self.delay_ticks)]
        self.snapshots[tick] = packet
        self.first = packet if self.first is None else self.first
        out = self.snapshots.get(tick - self.delay_ticks, self.first)
        self.snapshots.pop(tick - 64, None)
        return out

    def queue(self, packet, tick, initial):
        self.previous = initial if self.previous is None else self.previous
        due = tick + self.delay_ticks
        if self.implementation == 'A':
            heapq.heappush(self.heap, (due, tick, packet))
            while self.heap and self.heap[0][0] <= tick:
                self.previous = heapq.heappop(self.heap)[2]
        else:
            self.buckets.setdefault(due, []).append(packet)
            ready = sorted(key for key in self.buckets if key <= tick)
            for key in ready:
                self.previous = self.buckets.pop(key)[-1]
        return self.previous

    def adapter(self, packet):
        spec = self.node['nominal']
        scale, offset = float(spec.get('scale', 1)), float(spec.get('offset', 0))
        nominal = scale * packet.canonical() + offset
        if self.implementation == 'A':
            values = nominal * self.extra_gain
        else:
            values = (np.eye(len(nominal)) * self.extra_gain) @ nominal
        return Packet(values, packet.token, packet.channels, packet.state, scale, offset, packet.missing)

def merge_packets(packets):
    channels = tuple(sorted(c for packet in packets for c in packet.channels))
    if len(channels) != len(set(channels)):
        raise ValueError('merger channels overlap')
    values = np.zeros(14)
    state = packets[0].state
    digest = hashlib.blake2b(digest_size=16, person=b'cgx-merge-v1')
    for packet in sorted(packets, key=lambda p: p.channels):
        idx = list(packet.channels)
        if state:
            idx += [c + 7 for c in packet.channels]
        values[idx] = packet.canonical()[idx]
        digest.update(packet.token)
        digest.update(bytes(packet.channels))
    return Packet(values, digest.digest(), channels, state, missing=any(p.missing for p in packets))
