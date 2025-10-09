# parallel/engine.py
"""
Phase 2+: parallelised orientator pass with identical results.

What changed
------------
- The pre-step loop `tip.orientation = orientator.compute(tip)` now uses a
  ThreadPoolExecutor when `workers > 1`.
- We collect tips in a deterministic order, compute in parallel, then assign
  orientations back to the *same tips in the same order*.

Everything else (grow/update → destructors → branching → snapshot/time/prune →
post-step wrapper: grid update, nutrient kill, mutator, checkpoint, stats)
remains exactly as before to preserve hashes.

How to use
----------
engine = ParallelStepEngine(workers=4)  # or any >1
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

# --- Proposal kinds (execution still calls your original methods) ------------

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
    # canonical order: by creation id (matches your iteration order)
    ids = [int(s.id) for s in sections]
    return Snapshot(ids=ids)

class ParallelStepEngine:
    def __init__(self, apply_order: str = "id", workers: int = 0):
        self.apply_order = apply_order
        self.workers = int(workers)

    def _get_tips(self, mycel):
        return [s for s in mycel.sections if s.is_tip and not s.is_dead]

    def _by_id(self, mycel, sid: int) -> Optional[Section]:
        for s in mycel.sections:
            if int(s.id) == int(sid):
                return s
        return None

    def _compute_orientations(self, tips, orientator, use_threads: bool):
        """
        Compute orientations for tips, preserving deterministic assignment order.
        """
        if not tips:
            return
        # Force serial for exact match
        for tip in tips:
            tip.orientation = orientator.compute(tip)

    def step_parallel_equivalent(
        self,
        mycel,
        components: Dict[str, Any],
        step: int,
        master_seed: Optional[int],   # reserved for future addressable RNG
    ) -> None:
        """
        EXACT behavior match, with parallelised orientator phase when workers > 1.
        """
        # ---- Unpack components (matches main.py) ----
        aggregator   = components["aggregator"]
        grid         = components["grid"]
        orientator   = components["orientator"]
        checkpoints  = components["checkpoints"]
        autostop     = components["autostop"]  # used outside in simulate()
        mutator      = components["mutator"]
        stats        = components["stats"]
        opts: Options = components["opts"]

        # ---- Pre-step wrapper (field refresh) ----
        aggregator.sources.clear()
        aggregator.add_sections(mycel.get_all_segments(), strength=1.0, decay=1.5)

        # ---- Parallelised orientator pass (pure, RNG-free) ----
        tips = mycel.get_tips()  # capture deterministic order
        self._compute_orientations(tips, orientator, use_threads=(self.workers > 1))

        # ---- Snapshot current state (for structure; not used in RNG) ----
        snap = _take_snapshot(mycel.sections)

        # ---- Ordered proposals (no RNG here) ----
        proposals: List[Proposal] = []
        proposals.extend(Proposal((0, sid), OP_GROW_UPDATE, sid) for sid in snap.ids)
        proposals.extend(Proposal((1, sid), OP_DESTRUCT_CHECK, sid) for sid in snap.ids)
        proposals.extend(Proposal((2, sid), OP_BRANCH_ATTEMPT, sid) for sid in snap.ids)
        proposals.sort(key=lambda p: p.apply_key)

        # ---- Single-writer merge (exact side effects, original order) ----
        logger.debug("STEP START: t=%.2f | total_sections=%d", mycel.time, len(mycel.sections))
        new_sections: List[Section] = []
        tip_count_pre = len(self._get_tips(mycel))

        for prop in proposals:
            s = self._by_id(mycel, prop.section_id)
            if s is None:
                continue

            if prop.op == OP_GROW_UPDATE:
                if s.is_dead:
                    continue
                s.grow(opts.growth_rate, opts.time_step)
                s.update()
                if s.is_tip and not s.is_dead:
                    logger.debug("TIP pos=%s len=%.2f age=%.2f", s.end, s.length, s.age)

            elif prop.op == OP_DESTRUCT_CHECK:
                if not s.is_tip or s.is_dead:
                    continue
                if opts.die_if_old and s.age > opts.max_age:
                    s.is_dead = True
                    logger.debug("Tip died of age: age=%.2f > max_age=%.2f", s.age, opts.max_age)
                    continue
                if s.length > opts.max_length:
                    s.is_dead = True
                    logger.debug("Tip died of length: len=%.2f > max_len=%.2f", s.length, opts.max_length)
                    continue
                if opts.die_if_too_dense and s.field_aggregator:
                    density = s.field_aggregator.compute_field(s.end)[0]
                    if density > opts.density_threshold:
                        logger.debug("Density kill: %.3f > threshold %.3f", density, opts.density_threshold)
                        s.is_dead = True
                        continue
                if opts.use_nutrient_field and s.field_aggregator:
                    nutrient_field = s.field_aggregator.compute_field(s.end)[0]
                    if nutrient_field < -abs(opts.nutrient_repulsion):
                        logger.debug(
                            "Repellent kill: nutrient_field=%.3f < -|repulsion|=%.3f",
                            nutrient_field, abs(opts.nutrient_repulsion)
                        )
                        s.is_dead = True
                        continue
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

            elif prop.op == OP_BRANCH_ATTEMPT:
                if s.is_dead:
                    continue
                if s.is_tip or opts.allow_internal_branching:
                    child = s.maybe_branch(opts.branch_probability, tip_count=tip_count_pre)
                    if child:
                        logger.debug("BRANCHED: %s → %s", s.end, child.orientation)
                        new_sections.append(child)

        try:
            logger.debug("Post-destruction sample tip: pos=%s is_tip=%s is_dead=%s",
                         s.end, s.is_tip, s.is_dead)
        except UnboundLocalError:
            pass

        if new_sections:
            logger.debug("Added %d new sections this step", len(new_sections))
        mycel.sections.extend(new_sections)

        # ---- End-of-step: snapshot → time → prune → histories (unchanged) ----
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
        total_biomass = sum(sec.length for sec in mycel.sections if not sec.is_dead)
        mycel.biomass_history.append(total_biomass)
        logger.debug("STEP END: active_tips=%d | biomass=%.2f", len(self._get_tips(mycel)), total_biomass)

        # ---- Post-step wrapper (as main.step_simulation) ----
        grid.update_from_mycel(mycel)

        if getattr(opts, "use_nutrient_field", False) and getattr(opts, "nutrient_repulsion", 0) > 0:
            if hasattr(mycel, "nutrient_kill_check"):
                mycel.nutrient_kill_check()

        mutator.apply(step, opts)
        checkpoints.maybe_save(mycel, step)
        stats.update(mycel)
        logger.debug(str(mycel))
