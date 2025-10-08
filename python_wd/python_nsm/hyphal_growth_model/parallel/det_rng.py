# parallel/det_rng.py

"""
Deterministic RNG primitives for order-independent parallelism.

Usage:
    rng = DetRNG(master_seed=123)
    u = rng.u01(step=10, entity_id=42, k=0)    # uniform [0,1)
    z = rng.n01(step=10, entity_id=42, k=1)    # standard normal
    # For multiple draws per entity/step, increment k: 0,1,2,...
"""

from __future__ import annotations
import hashlib
import math

# map a 64-bit int to [0,1) with 53-bit mantissa (double precision)
_INV_2_53 = 1.0 / (1 << 53)

def _addr_hash(master_seed: int, step: int, entity_id: int, k: int) -> int:
    h = hashlib.sha256(f"{master_seed}:{step}:{entity_id}:{k}".encode()).digest()
    return int.from_bytes(h[:8], "little")  # 64-bit

def u01(master_seed: int, step: int, entity_id: int, k: int) -> float:
    x = _addr_hash(master_seed, step, entity_id, k)
    return ((x >> 11) & ((1 << 53) - 1)) * _INV_2_53

def n01(master_seed: int, step: int, entity_id: int, k: int) -> float:
    # Box–Muller from two u01 draws; fixed addressable k sequence
    u = max(u01(master_seed, step, entity_id, 2*k), 1e-16)
    v = u01(master_seed, step, entity_id, 2*k + 1)
    r = math.sqrt(-2.0 * math.log(u))
    return r * math.cos(2.0 * math.pi * v)

class DetRNG:
    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)

    def u01(self, step: int, entity_id: int, k: int) -> float:
        return u01(self.master_seed, step, entity_id, k)

    def n01(self, step: int, entity_id: int, k: int) -> float:
        return n01(self.master_seed, step, entity_id, k)
