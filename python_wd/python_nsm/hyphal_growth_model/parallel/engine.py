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
        """
        *** PLACEHOLDER ***
        Encapsulate per-section decisions (grow, update, branch, kill).
        This MUST be SIDE-EFFECT FREE. Use only the snapshot & opts to compute outputs.

        Next commit: transplant CSF operations here in the same call-count/RNG order.
        """
        proposals: List[Proposal] = []
        payloads: List[Any] = []
        # Example structure for later:
        # for sid in section_ids:
        #     row = snapshot.index_of[sid]
        #     k = 0
        #     u = rng.u01(step, sid, k); k += 1
        #     # derive decisions using snapshot.* arrays
        #     # payloads.append(...); proposals.append(Proposal((sid, sid), OP_..., payload_idx, target_id=sid))
        return proposals, payloads

    def step_parallel_equivalent(self, mycel, components: Dict[str, Any], step: int, master_seed: Optional[int]) -> None:
        """
        One deterministic step:
          1) Snapshot current state
          2) Compute proposals (optionally in parallel)
          3) Stable-sort proposals by apply_key to mirror CSF order
          4) Apply proposals serially to 'mycel' (single writer)
        """
        opts: Options = components["opts"]

        # --- robust seed handling (fix for None) ---
        # Prefer the explicit master_seed; else fall back to opts.seed; else constant 0
        eff_seed = master_seed if master_seed is not None else getattr(opts, "seed", None)
        if eff_seed is None:
            eff_seed = 0
        rng = DetRNG(eff_seed)

        # 1) Snapshot *before* any changes this step
        snapshot = take_snapshot(mycel.sections)

        # Partition section IDs for proposal computation
        section_ids = snapshot.ids[:]  # already in ascending ID

        # PHASE 1: serial proposal computation
        all_props, all_payloads = self.compute_proposals(mycel, snapshot, opts, rng, step, section_ids)

        # 3) Stable, deterministic order identical to CSF
        all_props.sort(key=lambda p: (p.apply_key[0], p.apply_key[1]))

        # 4) Apply serially (single writer). This is where we mutate 'mycel'.
        for prop in all_props:
            payload = all_payloads[prop.payload_idx] if prop.payload_idx is not None else None
            self._apply_proposal(mycel, opts, prop, payload)

        # While compute_proposals is a no-op, delegate to the CSF step to keep behaviour identical:
        if not all_props:
            import main as runner
            runner.step_simulation(mycel, components, step)

    # ---- Applying proposals (single-thread) ----

    def _apply_proposal(self, mycel, opts: Options, prop: Proposal, payload: Any) -> None:
        if prop.op == OP_UPDATE_SECTION:
            sid = prop.target_id
            s = next((sec for sec in mycel.sections if sec.id == sid), None)
            if s is None:
                return
            # payload should include deltas or new fields; placeholder:
            # s.length = payload.length; s.end = payload.end; etc.
            pass

        elif prop.op == OP_CREATE_BRANCH:
            # payload should contain all params to construct a new Section deterministically
            # e.g., (start_point, orientation, colour, parent_id)
            pass

        elif prop.op == OP_KILL_SECTION:
            sid = prop.target_id
            s = next((sec for sec in mycel.sections if sec.id == sid), None)
            if s is not None:
                s.is_dead = True

        else:
            # Unknown op — ignore
            pass
