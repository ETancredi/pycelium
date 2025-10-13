# parallel/engine.py
"""
Deterministic parallel step engine.

This version keeps Orientator serial (to preserve RNG/state order), but
parallelises heavy, read-only field evaluations using a persistent thread pool.
Order-sensitive parts (grow/update merge, destructor semantics, branching)
remain serial and in the same order as the single-thread engine.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

from core.options import Options
from core.section import Section

logger = logging.getLogger("pycelium.parallel.engine")

OP_GROW_UPDATE    = 0
OP_DESTRUCT_CHECK = 1
OP_BRANCH_ATTEMPT = 2

@dataclass(frozen=True)
class Proposal:
    apply_key: Tuple[int, int]  # (pass_order, section_id)
    op: int
    section_id: int

@dataclass
class Snapshot:
    ids: List[int]

def _take_snapshot(sections: List[Section]) -> Snapshot:
    ids = [int(s.id) for s in sections]
    return Snapshot(ids=ids)

class ParallelStepEngine:
    def __init__(self, apply_order: str = "id", workers: int = 0):
        self.apply_order = apply_order
        self.workers = int(workers)
        # NEW: persistent pool (created once, reused every step)
        self._pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

    def shutdown(self):
        """Cleanly shut down the persistent pool (call after the run)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _get_tips(self, mycel):
        return [s for s in mycel.sections if s.is_tip and not s.is_dead]

    def _by_id(self, mycel, sid: int) -> Optional[Section]:
        for s in mycel.sections:
            if int(s.id) == int(sid):
                return s
        return None

    def _compute_orientations_serial(self, tips, orientator):
        # Keep serial to preserve exact RNG/state sequencing
        for tip in tips:
            tip.orientation = orientator.compute(tip)

    def step_parallel_equivalent(
        self,
        mycel,
        components: Dict[str, Any],
        step: int,
        master_seed: Optional[int],
    ) -> None:
        aggregator   = components["aggregator"]
        grid         = components["grid"]
        orientator   = components["orientator"]
        checkpoints  = components["checkpoints"]
        autostop     = components["autostop"]
        mutator      = components["mutator"]
        stats        = components["stats"]
        opts: Options = components["opts"]

        # Pre-step wrapper (field refresh)
        aggregator.sources.clear()
        aggregator.add_sections(mycel.get_all_segments(), strength=1.0, decay=1.5)

        # Orientations (kept serial)
        tips_for_orient = self._get_tips(mycel)
        self._compute_orientations_serial(tips_for_orient, orientator)

        # Snapshot for ordered iteration
        snap = _take_snapshot(mycel.sections)

        # Build ordered proposals (preserve exact application order)
        proposals: List[Proposal] = []
        proposals.extend(Proposal((0, sid), OP_GROW_UPDATE, sid) for sid in snap.ids)
        proposals.extend(Proposal((1, sid), OP_DESTRUCT_CHECK, sid) for sid in snap.ids)
        proposals.extend(Proposal((2, sid), OP_BRANCH_ATTEMPT, sid) for sid in snap.ids)
        proposals.sort(key=lambda p: p.apply_key)

        # ---- Merge: pass 0 (grow/update) ----
        logger.debug("STEP START: t=%.2f | total_sections=%d", mycel.time, len(mycel.sections))
        new_sections: List[Section] = []
        for prop in proposals:
            if prop.op != OP_GROW_UPDATE:
                continue
            s = self._by_id(mycel, prop.section_id)
            if s is None or s.is_dead:
                continue
            s.grow(opts.growth_rate, opts.time_step)
            s.update()
            if s.is_tip and not s.is_dead:
                logger.debug("TIP pos=%s len=%.2f age=%.2f", s.end, s.length, s.age)

        # ---- Threaded precompute of fields for destructor checks (reusing pool) ----
        # We precompute AFTER grow/update (to match end-positions used in the checks).
        need_density  = bool(getattr(opts, "die_if_too_dense", False))
        need_nutrient = bool(getattr(opts, "use_nutrient_field", False) and getattr(opts, "nutrient_repulsion", 0) != 0)

        current_tips = self._get_tips(mycel)
        tip_sids     = [int(t.id) for t in current_tips]

        density_map: Dict[int, float] = {}
        nutrient_map: Dict[int, float] = {}

        def _field_at_end(t: Section) -> float:
            agg = t.field_aggregator
            return agg.compute_field(t.end)[0] if agg else 0.0

        if (need_density or need_nutrient) and current_tips:
            if self._pool is not None:
                values = list(self._pool.map(_field_at_end, current_tips))
            else:
                values = [_field_at_end(t) for t in current_tips]

            for sid, val in zip(tip_sids, values):
                if need_density:
                    density_map[sid] = val
                if need_nutrient:
                    nutrient_map[sid] = val

        # ---- Merge: pass 1 (destructor checks) ----
        for prop in proposals:
            if prop.op != OP_DESTRUCT_CHECK:
                continue
            s = self._by_id(mycel, prop.section_id)
            if s is None or not s.is_tip or s.is_dead:
                continue

            # A) age
            if opts.die_if_old and s.age > opts.max_age:
                s.is_dead = True
                logger.debug("Tip died of age: age=%.2f > max_age=%.2f", s.age, opts.max_age)
                continue

            # B) length
            if s.length > opts.max_length:
                s.is_dead = True
                logger.debug("Tip died of length: len=%.2f > max_len=%.2f", s.length, opts.max_length)
                continue

            # C) density (use precomputed value if requested)
            if need_density and s.field_aggregator:
                dens = density_map.get(int(s.id), s.field_aggregator.compute_field(s.end)[0])
                if dens > opts.density_threshold:
                    logger.debug("Density kill: %.3f > threshold %.3f", dens, opts.density_threshold)
                    s.is_dead = True
                    continue

            # D) nutrient repulsion (use precomputed value if requested)
            if need_nutrient and s.field_aggregator:
                nf = nutrient_map.get(int(s.id), s.field_aggregator.compute_field(s.end)[0])
                if nf < -abs(opts.nutrient_repulsion):
                    logger.debug(
                        "Repellent kill: nutrient_field=%.3f < -|repulsion|=%.3f",
                        nf, abs(opts.nutrient_repulsion)
                    )
                    s.is_dead = True
                    continue

            # E) isolation (left serial & live against current tips to preserve dynamics)
            if s.field_aggregator:
                nearby = 0
                for other in self._get_tips(mycel):
                    if other is s:
                        continue
                    if s.end.distance_to(other.end) <= opts.neighbour_radius:
                        nearby += 1
                if nearby < opts.min_supported_tips:
                    logger.debug(
                        "Isolation kill: neighbours=%d < min_supported=%d",
                        nearby, opts.min_supported_tips
                    )
                    s.is_dead = True
                    continue

        # Post-destruction sample log (unchanged)
        try:
            logger.debug("Post-destruction sample tip: pos=%s is_tip=%s is_dead=%s",
                         s.end, s.is_tip, s.is_dead)
        except UnboundLocalError:
            pass

        # ---- Merge: pass 2 (branch attempts; RNG lives here) ----
        tip_count_pre = len(self._get_tips(mycel))  # as in the previous version
        for prop in proposals:
            if prop.op != OP_BRANCH_ATTEMPT:
                continue
            s = self._by_id(mycel, prop.section_id)
            if s is None or s.is_dead:
                continue
            if s.is_tip or opts.allow_internal_branching:
                child = s.maybe_branch(opts.branch_probability, tip_count=tip_count_pre)
                if child:
                    logger.debug("BRANCHED: %s → %s", s.end, child.orientation)
                    new_sections.append(child)

        if new_sections:
            logger.debug("Added %d new sections this step", len(new_sections))
        mycel.sections.extend(new_sections)

        # ---- End-of-step unchanged: snapshot → time → prune → histories ----
        step_snapshot = [
            {
                "time": mycel.time,
                "x": tip.end.coords[0],
                "y": tip.end.coords[1],
                "z": tip.end.coords[2],
                "age": tip.age,
                "length": tip.length,
            }
            for tip in self._get_tips(mycel)
        ]
        mycel.time_series.append(step_snapshot)

        mycel.time += opts.time_step

        if hasattr(opts, "max_supported_tips") and opts.max_supported_tips > 0:
            active_tips = self._get_tips(mycel)
            if len(active_tips) > opts.max_supported_tips:
                logger.info(
                    "Tip pruning: %d tips exceed max (%d) → pruning",
                    len(active_tips), opts.max_supported_tips
                )
                excess = len(active_tips) - opts.max_supported_tips
                to_prune = np.random.choice(active_tips, size=excess, replace=False)
                for tip in to_prune:
                    tip.is_dead = True
                    logger.debug("Pruned tip at %s due to overcrowding", tip.end)

        tip_data = [
            (tip.end.coords[0], tip.end.coords[1], tip.end.coords[2])
            for tip in self._get_tips(mycel)
        ]
        mycel.step_history.append((mycel.time, tip_data))
        total_biomass = sum(sec.length for sec in mycel.sections if not s.is_dead)  # keep as-is to preserve outputs
        mycel.biomass_history.append(total_biomass)
        logger.debug("STEP END: active_tips=%d | biomass=%.2f", len(self._get_tips(mycel)), total_biomass)

        # Post-step wrapper (as in main.step_simulation)
        grid.update_from_mycel(mycel)
        if getattr(opts, "use_nutrient_field", False) and getattr(opts, "nutrient_repulsion", 0) > 0:
            if hasattr(mycel, "nutrient_kill_check"):
                mycel.nutrient_kill_check()
        mutator.apply(step, opts)
        checkpoints.maybe_save(mycel, step)
        stats.update(mycel)
        logger.debug(str(mycel))
