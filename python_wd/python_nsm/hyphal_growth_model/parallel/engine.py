# parallel/engine.py
# parallel/engine.py
"""
Deterministic parallel step skeleton for Pycelium.

Phase 1 (this file): behaviour-identical reimplementation of Mycel.step()
inside the engine, so we no longer fall back to runner.step_simulation.
We keep the same visit order and RNG usage (np.random / random) to
match your current CSF traces exactly.

After we confirm hashes match, we'll parallelise safe phases (e.g., the
per-section grow/update and branch decision *proposal* stage) while
keeping a single-writer, stable-order merge.

Nothing here mutates global state outside the intended, CSF-like sequence.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import os
import random
import numpy as np

from core.section import Section
from core.options import Options
from core.point import MPoint  # only for type hints

# ---- Proposal API (kept for Phase 2; unused in Phase 1) ---------------------

OP_UPDATE_SECTION = 1
OP_CREATE_BRANCH  = 2
OP_KILL_SECTION   = 3

@dataclass(frozen=True)
class Proposal:
    apply_key: Tuple[int, int]        # (csf_order_key, section_id)
    op: int                           # one of OP_*
    payload_idx: Optional[int]        # index into payloads list for data (None if unused)
    target_id: Optional[int]          # section id target (if applicable)

@dataclass
class Snapshot:
    ids: List[int]
    is_tip: List[bool]
    is_dead: List[bool]

# -----------------------------------------------------------------------------

def _effective_seed(opts: Options, master_seed: Optional[int]) -> int:
    s = master_seed if master_seed is not None else getattr(opts, "seed", None)
    return 0 if s is None else int(s)

class ParallelStepEngine:
    """
    Deterministic engine that mirrors Mycel.step() exactly (Phase 1).
    After validation against your hash oracle, we'll enable parallel
    proposal computation in Phase 2.
    """
    def __init__(self, apply_order: str = "id", workers: int = 0):
        self.apply_order = apply_order
        self.workers = int(workers)

    # --- Phase 1: direct, behaviour-identical step implementation ---

    def step_parallel_equivalent(self, mycel, components: Dict[str, Any], step: int, master_seed: Optional[int]) -> None:
        """
        EXACT mirror of core/mycel.py::Mycel.step(), using the same loops,
        order, RNG calls, and side-effects — so per-step hashes remain identical.
        """
        opts: Options = components["opts"]

        # Keep RNG usage identical to CSF run
        # (We seed once at simulation setup; here we just use np.random/random
        #  exactly as in Mycel.step(), so the draw sequence matches.)
        new_sections: List[Section] = []

        # Diagnostic parity with your Mycel.step (optional)
        # logger.debug("STEP START: t=%.2f | total_sections=%d", mycel.time, len(mycel.sections))

        tip_count = len([s for s in mycel.sections if s.is_tip and not s.is_dead])

        # 1) Grow & update existing sections (same list order)
        for section in mycel.sections:
            if section.is_dead:
                continue
            section.grow(opts.growth_rate, opts.time_step)
            section.update()

        # 2) Destructor logic: age, length, density, nutrient, isolation
        for section in mycel.sections:
            if not section.is_tip or section.is_dead:
                continue

            # A) age
            if getattr(opts, "die_if_old", False) and section.age > opts.max_age:
                section.is_dead = True
                continue

            # B) length
            if section.length > opts.max_length:
                section.is_dead = True
                continue

            # C) density (uses field_aggregator if present)
            if getattr(opts, "die_if_too_dense", False) and section.field_aggregator:
                density = section.field_aggregator.compute_field(section.end)[0]
                if density > opts.density_threshold:
                    section.is_dead = True
                    continue

            # D) nutrient repulsion kill
            if getattr(opts, "use_nutrient_field", False) and section.field_aggregator:
                nutrient_field = section.field_aggregator.compute_field(section.end)[0]
                if nutrient_field < -abs(getattr(opts, "nutrient_repulsion", 0.0)):
                    section.is_dead = True
                    continue

            # E) isolation
            if section.field_aggregator:
                nearby_count = 0
                # neighbour check only among *current* tips
                for other in [s for s in mycel.sections if s.is_tip and not s.is_dead]:
                    if other is section:
                        continue
                    if section.end.distance_to(other.end) <= opts.neighbour_radius:
                        nearby_count += 1
                if nearby_count < opts.min_supported_tips:
                    section.is_dead = True
                    continue

        # 3) Branching
        for section in mycel.sections:
            if section.is_dead:
                continue
            if section.is_tip or opts.allow_internal_branching:
                child = section.maybe_branch(opts.branch_probability, tip_count=tip_count)
                if child:
                    new_sections.append(child)

        if new_sections:
            mycel.sections.extend(new_sections)

        # 4) Record tip snapshot (positions and metrics)
        step_snapshot = [
            {
                "time": mycel.time,
                "x": tip.end.coords[0],
                "y": tip.end.coords[1],
                "z": tip.end.coords[2],
                "age": tip.age,
                "length": tip.length,
            }
            for tip in [s for s in mycel.sections if s.is_tip and not s.is_dead]
        ]
        mycel.time_series.append(step_snapshot)

        # Advance time
        mycel.time += opts.time_step

        # 5) Pruning by max_supported_tips
        if hasattr(opts, "max_supported_tips") and opts.max_supported_tips > 0:
            active_tips = [s for s in mycel.sections if s.is_tip and not s.is_dead]
            if len(active_tips) > opts.max_supported_tips:
                excess = len(active_tips) - opts.max_supported_tips
                to_prune = np.random.choice(active_tips, size=excess, replace=False)
                for tip in to_prune:
                    tip.is_dead = True

        # 6) Update histories / biomass
        tip_data = [(tip.end.coords[0], tip.end.coords[1], tip.end.coords[2])
                    for tip in [s for s in mycel.sections if s.is_tip and not s.is_dead]]
        mycel.step_history.append((mycel.time, tip_data))
        total_biomass = sum(sec.length for sec in mycel.sections if not sec.is_dead)
        mycel.biomass_history.append(total_biomass)

    # --- Phase 2 scaffolding (kept for later parallelisation) ----------------

    def compute_proposals(
        self,
        mycel,
        snapshot: Snapshot,
        opts: Options,
        step: int,
        section_ids: List[int],
    ) -> Tuple[List[Proposal], List[Any]]:
        """
        Placeholder for Phase 2: order-independent proposal generation.
        In Phase 1 we do a direct, mirrored step (above).
        """
        return [], []

    def _apply_proposal(self, mycel, opts: Options, prop: Proposal, payload: Any) -> None:
        # Phase 2 only
        pass
