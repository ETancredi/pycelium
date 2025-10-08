# parallel/engine.py
"""
Deterministic engine: faithful, no-delegation mirror of core/mycel.py::Mycel.step().

This replays your single-thread step sequence against the same objects in the same
order (grow → update → destructors → branching → tip snapshot → time advance → pruning → histories).
No new RNG calls are introduced; we rely on the exact np.random/random usage in your
Section.maybe_branch and the pruning step to keep per-step hashes identical.

Phase 2 (after we confirm hash equality): refactor this into
snapshot → propose → stable-merge to enable parallelism without changing hashes.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import logging

from core.section import Section
from core.options import Options
from core.point import MPoint  # only for annotations/log formatting

logger = logging.getLogger("pycelium.parallel.engine")

class ParallelStepEngine:
    """
    Phase 1: exact behavior transplant of Mycel.step(), no delegation.
    """

    def __init__(self, apply_order: str = "id", workers: int = 0):
        # (apply_order/workers are for Phase 2; unused in this faithful mirror)
        self.apply_order = apply_order
        self.workers = int(workers)

    # --- helpers (mirrors mycel.get_tips to keep this file self-contained) ---
    def _get_tips(self, mycel):
        return [s for s in mycel.sections if s.is_tip and not s.is_dead]

    # --- faithful step implementation ----------------------------------------
    def step_parallel_equivalent(
        self,
        mycel,
        components: Dict[str, Any],
        step: int,
        master_seed: Optional[int],  # kept for Phase 2; unused here
    ) -> None:
        """
        EXACT mirror of core/mycel.py::Mycel.step(), using your uploaded code.
        """

        opts: Options = components["opts"]
        new_sections: List[Section] = []

        # Step start (debug-only, matches your logging)
        logger.debug("STEP START: t=%.2f | total_sections=%d", mycel.time, len(mycel.sections))

        # Count active tips BEFORE growth (this is passed to maybe_branch)
        tip_count = len(self._get_tips(mycel))

        # 1) Grow & update existing sections (same list order)
        for section in mycel.sections:
            if section.is_dead:
                continue
            # Grow by growth_rate over time_step
            section.grow(opts.growth_rate, opts.time_step)
            # Update internal state (e.g., exact length recompute)
            section.update()
            # Debug trace for living tips
            if section.is_tip and not section.is_dead:
                logger.debug("TIP pos=%s len=%.2f age=%.2f", section.end, section.length, section.age)

        # 2) Destructor logic: check *alive tips* only
        for section in mycel.sections:
            if not section.is_tip or section.is_dead:
                continue

            # A) age limit
            if opts.die_if_old and section.age > opts.max_age:
                section.is_dead = True
                logger.debug("Tip died of age: age=%.2f > max_age=%.2f", section.age, opts.max_age)
                continue

            # B) length limit
            if section.length > opts.max_length:
                section.is_dead = True
                logger.debug("Tip died of length: len=%.2f > max_len=%.2f", section.length, opts.max_length)
                continue

            # C) density kill (if aggregator present)
            if opts.die_if_too_dense and section.field_aggregator:
                density = section.field_aggregator.compute_field(section.end)[0]
                if density > opts.density_threshold:
                    logger.debug("Density kill: %.3f > threshold %.3f", density, opts.density_threshold)
                    section.is_dead = True
                    continue

            # D) nutrient repulsion kill
            if opts.use_nutrient_field and section.field_aggregator:
                nutrient_field = section.field_aggregator.compute_field(section.end)[0]
                if nutrient_field < -abs(opts.nutrient_repulsion):
                    logger.debug(
                        "Repellent kill: nutrient_field=%.3f < -|repulsion|=%.3f",
                        nutrient_field, abs(opts.nutrient_repulsion)
                    )
                    section.is_dead = True
                    continue

            # E) isolation (neighbour count within radius)
            if section.field_aggregator:
                nearby_count = 0
                for other in self._get_tips(mycel):
                    if other is section:
                        continue
                    if section.end.distance_to(other.end) <= opts.neighbour_radius:
                        nearby_count += 1
                if nearby_count < opts.min_supported_tips:
                    logger.debug(
                        "Isolation kill: neighbours=%d < min_supported=%d",
                        nearby_count, opts.min_supported_tips
                    )
                    section.is_dead = True
                    continue

        # Optional log of last-checked section (exactly as in your code)
        try:
            logger.debug(
                "Post-destruction sample tip: pos=%s is_tip=%s is_dead=%s",
                section.end, section.is_tip, section.is_dead
            )
        except UnboundLocalError:
            pass

        # 3) Branching: tips (or internal if allowed)
        for section in mycel.sections:
            if section.is_dead:
                continue
            if section.is_tip or opts.allow_internal_branching:
                child = section.maybe_branch(opts.branch_probability, tip_count=tip_count)
                if child:
                    logger.debug("BRANCHED: %s → %s", section.end, child.orientation)
                    new_sections.append(child)

        # Add newly created sections
        if new_sections:
            logger.debug("Added %d new sections this step", len(new_sections))
        mycel.sections.extend(new_sections)

        # 4) Snapshot of current tips (positions & metrics) at current time
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

        # Advance time
        mycel.time += opts.time_step

        # 5) Optional pruning by max_supported_tips
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

        # 6) Update history and biomass
        tip_data = [
            (tip.end.coords[0], tip.end.coords[1], tip.end.coords[2])
            for tip in self._get_tips(mycel)
        ]
        mycel.step_history.append((mycel.time, tip_data))
        total_biomass = sum(sec.length for sec in mycel.sections if not sec.is_dead)
        mycel.biomass_history.append(total_biomass)
        logger.debug("STEP END: active_tips=%d | biomass=%.2f", len(self._get_tips(mycel)), total_biomass)
