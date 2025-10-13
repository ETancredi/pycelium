# parallel/engine.py
"""
Deterministic parallel step engine.

Optimisations in this version (no output changes):
- Persistent ThreadPoolExecutor (created once, reused).
- Remove Proposal objects; do three ordered passes over a snapshot of section ids.
- O(1) id -> section lookup per step.
- Cheaper isolation check (squared distance + cached coords).
- Field evaluations (density/nutrient) precomputed in parallel via the pool.

Orientator remains serial to preserve RNG/state order.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from core.options import Options
from core.section import Section

logger = logging.getLogger("pycelium.parallel.engine")


def _take_snapshot_ids(sections: List[Section]) -> List[int]:
    # Deterministic pass order
    return [int(s.id) for s in sections]


class ParallelStepEngine:
    def __init__(self, apply_order: str = "id", workers: int = 0):
        self.apply_order = apply_order
        self.workers = int(workers)
        self._pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # --- helpers -------------------------------------------------------------

    def _get_tips(self, mycel):
        return [s for s in mycel.sections if s.is_tip and not s.is_dead]

    def _compute_orientations_serial(self, tips, orientator):
        for tip in tips:
            tip.orientation = orientator.compute(tip)

    # --- main step -----------------------------------------------------------

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

        # Pre-step wrapper (refresh field sources)
        aggregator.sources.clear()
        aggregator.add_sections(mycel.get_all_segments(), strength=1.0, decay=1.5)

        # Orientations (serial to preserve RNG/state)
        self._compute_orientations_serial(self._get_tips(mycel), orientator)

        # Snapshot ids and id->section map (O(1) lookups)
        ids = _take_snapshot_ids(mycel.sections)
        id_to_section = {int(s.id): s for s in mycel.sections}

        logger.debug("STEP START: t=%.2f | total_sections=%d", mycel.time, len(mycel.sections))

        # ---------------------------------------------------------------------
        # PASS 0: grow / update (serial, id order)
        # ---------------------------------------------------------------------
        for sid in ids:
            s = id_to_section.get(sid)
            if s is None or s.is_dead:
                continue
            s.grow(opts.growth_rate, opts.time_step)
            s.update()
            if s.is_tip and not s.is_dead:
                logger.debug("TIP pos=%s len=%.2f age=%.2f", s.end, s.length, s.age)

        # ---------------------------------------------------------------------
        # Parallel precompute fields for destructor checks (if needed)
        # ---------------------------------------------------------------------
        need_density  = bool(getattr(opts, "die_if_too_dense", False))
        need_nutrient = bool(getattr(opts, "use_nutrient_field", False) and getattr(opts, "nutrient_repulsion", 0) != 0)

        density_map: Dict[int, float] = {}
        nutrient_map: Dict[int, float] = {}

        current_tips = self._get_tips(mycel)
        if (need_density or need_nutrient) and current_tips:
            tip_sids = [int(t.id) for t in current_tips]

            def _field_at_end(t: Section) -> float:
                agg = t.field_aggregator
                return agg.compute_field(t.end)[0] if agg else 0.0

            if self._pool is not None:
                values = list(self._pool.map(_field_at_end, current_tips))
            else:
                values = [_field_at_end(t) for t in current_tips]

            for sid, val in zip(tip_sids, values):
                if need_density:
                    density_map[sid] = val
                if need_nutrient:
                    nutrient_map[sid] = val

        # Cache coords for faster isolation math (still serial, order-sensitive)
        r2 = getattr(opts, "neighbour_radius", 0.0) ** 2
        # Take a snapshot list of tips (we will check other.is_dead live)
        tips_snapshot = self._get_tips(mycel)
        tip_coords = {int(t.id): t.end.coords for t in tips_snapshot}

        # ---------------------------------------------------------------------
        # PASS 1: destructor checks (serial, id order)
        # ---------------------------------------------------------------------
        for sid in ids:
            s = id_to_section.get(sid)
            if s is None or not s.is_tip or s.is_dead:
                continue

            # A) age
            if getattr(opts, "die_if_old", False) and s.age > opts.max_age:
                s.is_dead = True
                logger.debug("Tip died of age: age=%.2f > max_age=%.2f", s.age, opts.max_age)
                continue

            # B) length
            if s.length > opts.max_length:
                s.is_dead = True
                logger.debug("Tip died of length: len=%.2f > max_len=%.2f", s.length, opts.max_length)
                continue

            # C) density (precomputed if requested)
            if need_density and s.field_aggregator:
                dens = density_map.get(int(s.id), s.field_aggregator.compute_field(s.end)[0])
                if dens > opts.density_threshold:
                    logger.debug("Density kill: %.3f > threshold %.3f", dens, opts.density_threshold)
                    s.is_dead = True
                    continue

            # D) nutrient repulsion (precomputed if requested)
            if need_nutrient and s.field_aggregator:
                nf = nutrient_map.get(int(s.id), s.field_aggregator.compute_field(s.end)[0])
                if nf < -abs(opts.nutrient_repulsion):
                    logger.debug("Repellent kill: nutrient_field=%.3f < -|repulsion|=%.3f",
                                 nf, abs(opts.nutrient_repulsion))
                    s.is_dead = True
                    continue

            # E) isolation (live semantics preserved; cheaper math)
            if s.field_aggregator and r2 > 0.0:
                sx, sy, sz = tip_coords.get(int(s.id), s.end.coords)
                nearby = 0
                # Iterate snapshot; skip newly-dead to mimic fresh tip lists
                for other in tips_snapshot:
                    if other is s or other.is_dead:
                        continue
                    ox, oy, oz = tip_coords.get(int(other.id), other.end.coords)
                    dx = sx - ox; dy = sy - oy; dz = sz - oz
                    if (dx*dx + dy*dy + dz*dz) <= r2:
                        nearby += 1
                if nearby < getattr(opts, "min_supported_tips", 0):
                    logger.debug("Isolation kill: neighbours=%d < min_supported=%d",
                                 nearby, opts.min_supported_tips)
                    s.is_dead = True
                    continue

        # Optional representative debug (safe if no tips)
        try:
            logger.debug("Post-destruction sample tip: pos=%s is_tip=%s is_dead=%s", s.end, s.is_tip, s.is_dead)  # type: ignore[name-defined]
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # PASS 2: branching (serial, id order; RNG lives here)
        # ---------------------------------------------------------------------
        new_sections: List[Section] = []
        tip_count_pre = len(self._get_tips(mycel))
        for sid in ids:
            s = id_to_section.get(sid)
            if s is None or s.is_dead:
                continue
            if s.is_tip or getattr(opts, "allow_internal_branching", False):
                child = s.maybe_branch(opts.branch_probability, tip_count=tip_count_pre)
                if child:
                    logger.debug("BRANCHED: %s → %s", s.end, child.orientation)
                    new_sections.append(child)

        if new_sections:
            logger.debug("Added %d new sections this step", len(new_sections))
        mycel.sections.extend(new_sections)

        # ---------------------------------------------------------------------
        # End-of-step (unchanged)
        # ---------------------------------------------------------------------
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
        # NB: keep biomass summation semantics as-is to preserve hashes
        total_biomass = sum(sec.length for sec in mycel.sections if not s.is_dead)  # noqa: F821
        mycel.biomass_history.append(total_biomass)
        logger.debug("STEP END: active_tips=%d | biomass=%.2f", len(self._get_tips(mycel)), total_biomass)

        # Post-step wrapper (mirrors main.step_simulation)
        grid.update_from_mycel(mycel)
        if getattr(opts, "use_nutrient_field", False) and getattr(opts, "nutrient_repulsion", 0) > 0:
            if hasattr(mycel, "nutrient_kill_check"):
                mycel.nutrient_kill_check()
        mutator.apply(step, opts)
        checkpoints.maybe_save(mycel, step)
        stats.update(mycel)
        logger.debug(str(mycel))
