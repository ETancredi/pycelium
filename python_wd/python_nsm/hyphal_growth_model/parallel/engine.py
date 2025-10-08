# parallel/engine.py

"""
Deterministic parallel step skeleton for Pycelium.

Phase 0 (this file): structure and determinism only.
- Takes a read-only snapshot of state.
- Computes per-entity proposals (in parallel in Phase 2).
- Applies proposals in a *stable, CSF-identical order*.
- Uses counter-based RNG keyed by (seed, step, entity_id, k).

Phase 1: transplant your CSF step logic into `compute_proposals`.
Phase 2: turn on threads for the proposals stage.

Nothing here mutates global state outside the stable-merge section.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import os

from parallel.det_rng import DetRNG
from core.section import Section
from core.options import Options

# ---- Proposal model ---------------------------------------------------------

OP_UPDATE_SECTION = 1    # grow/update fields on an existing Section
OP_CREATE_BRANCH  = 2    # create a new Section as a child
OP_KILL_SECTION   = 3    # mark section dead

@dataclass(frozen=True)
class Proposal:
    apply_key: Tuple[int, int]   # (csf_order_key, section_id)
    op: int                      # one of OP_* above
    payload_idx: Optional[int]   # index into payloads list for data (None if unused)
    target_id: Optional[int]     # section being updated/branched/culled; None if not applicable

# ---- Snapshot model ---------------------------------------------------------

@dataclass
class Snapshot:
    ids: List[int]
    is_tip: List[bool]
    is_dead: List[bool]
    start_xyz: List[Tuple[float, float, float]]
    end_xyz:   List[Tuple[float, float, float]]
    orient_xyz:List[Tuple[float, float, float]]
    ages: List[float]
    lengths: List[float]
    index_of: Dict[int, int]  # section_id -> row idx

def _pt_xyz(p) -> Tuple[float, float, float]:
    if hasattr(p, "coords"):
        c = p.coords
        return float(c[0]), float(c[1]), float(c[2])
    # legacy fallback
    return float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0))

def take_snapshot(sections: List[Section]) -> Snapshot:
    # canonical order: ascending section.id
    ordered = sorted(sections, key=lambda s: int(s.id))
    ids, is_tip, is_dead = [], [], []
    start_xyz, end_xyz, orient_xyz = [], [], []
    ages, lengths = [], []
    index_of = {}
    for row, s in enumerate(ordered):
        ids.append(int(s.id))
        is_tip.append(bool(s.is_tip))
        is_dead.append(bool(s.is_dead))
        start_xyz.append(_pt_xyz(s.start))
        end_xyz.append(_pt_xyz(s.end))
        orient_xyz.append(_pt_xyz(s.orientation))
        ages.append(float(getattr(s, "age", 0.0)))
        lengths.append(float(getattr(s, "length", 0.0)))
        index_of[int(s.id)] = row
    return Snapshot(ids, is_tip, is_dead, start_xyz, end_xyz, orient_xyz, ages, lengths, index_of)

# ---- Core engine ------------------------------------------------------------

class ParallelStepEngine:
    """
    Deterministic proposals → stable-merge engine.

    Configuration:
      - apply_order = "id" (default) — matches CSF if the loop visits sections by creation id.
      - workers: number of threads used at the proposals stage (Phase 2).
    """
    def __init__(self, apply_order: str = "id", workers: int = 0):
        self.apply_order = apply_order
        self.workers = int(workers)

    @staticmethod
    def _apply_key_for(section: Section) -> Tuple[int, int]:
        # First key is the deterministic CSF order (here: section.id), second is id to keep total order.
        return (int(section.id), int(section.id))

    def compute_proposals(
        self,
        mycel,
        snapshot: Snapshot,
        opts: Options,
        rng: DetRNG,
        step: int,
        section_ids: List[int],
    ) -> Tuple[List[Proposal], List[Any]]:
