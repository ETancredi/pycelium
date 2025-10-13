# parallel/engine.py
"""
Deterministic parallel step engine with lightweight timing instrumentation.

Optimisations already included (no output changes):
- Persistent ThreadPoolExecutor (created once, reused).
- Three ordered passes over a snapshot of section ids (no Proposal objects).
- O(1) id -> section lookup per step.
- Cheaper isolation check (squared distance + cached coords).
- Field evaluations (density/nutrient) precomputed via the persistent pool.
- (Optional) Parallelised orientator pass with deterministic assignment.

Set env PYCELIUM_TIMINGS=1 to print a per-phase timing summary at shutdown.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging, os, time
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

        # --- timing accumulators (seconds) ---
        self._t_orient = 0.0
        self._t_grow = 0.0
        self._t_fields = 0.0
        self._t_destruct = 0.0
        self._t_branch = 0.0
        self._t_post = 0.0
        self._steps = 0

    def shutdown(self):
        """Cleanly shut down the persistent pool (call after the run)."""
        if os.getenv("PYCELIUM_TIMINGS", "").lower() in ("1", "true", "yes"):
            total = self._t_orient + self._t_grow + self._t_fields + self._t_destruct + self._t_branch + self._t_post
            def pct(x): return (100.0 * x / total) if total > 0 else 0.0
            print("\n⏱️  Parallel engine phase timings (aggregate):")
            print(f"  steps:           {self._steps}")
            print(f"  orientator:      {self._t_orient:8.3f} s  ({pct(self._t_orient):5.1f}%)")
            print(f"  grow/update:     {self._t_grow:8.3f} s  ({pct(self._t_grow):5.1f}%)")
            print(f"  field precomp:   {self._t_fields:8.3f} s  ({pct(self._t_fields):5.1f}%)")
            print(f"  destructors:     {self._t_destruct:8.3f} s  ({pct(self._t_destruct):5.1f}%)")
            print(f"  branching:       {self._t_branch:8.3f} s  ({pct(self._t_branch):5.1f}%)")
            print(f"  post-step wrap:  {self._t_post:8.3f} s  ({pct(self._t_post):5.1f}%)")
            print(f"  total (tracked): {total:8.3f} s")
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # --- helpers -------------------------------------------------------------

    def _get_tips(self, mycel):
        return [s for s in mycel.sections if s.is_tip and not s.is_dead]

    def _compute_orientations(self, tips: List[Section], orientator, opts: Options):
        """
        Compute orientations for all tips.
        If opts.parallelise_orientator and workers>1, do it in a pool but
        assign results back in the same order for determinism.
        """
        if not tips:
            return

        # Always use a stable ordering (list preserves current traversal order)
        ordered = list(tips)

        parallel_ok = (
            self._pool is not None and
            bool(getattr(opts, "parallelise_orientator", False))
        )

        if parallel_ok:
            # Parallel compute; deterministic assign
            results = list(self._pool.map(orientator.compute, ordered))
            for tip, ori in zip(ordered, results):
                tip.orientation = ori
        else:
            # Serial fallback
            for tip in ordered:
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

        # -------- ORIENTATIONS (optionally parallel) --------
        t_or = time.perf_counter()
        self._compute_orientations(self._get_tips(mycel), orientator, opts)
        self._t_orient += (time.perf_counter() - t_or)

        # Snapshot ids and id->section map (O(1) lookups)
        ids = _take_snapshot_ids(mycel.sections)
        id_to_section = {int(s.id): s for s in mycel.sections}

        logger.debug("STEP START: t=%.2f | total_sections=%d", mycel.time, len(mycel.sections))

        # ---------------------------------------------------------------------
        # PASS 0: grow / update
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        for sid in ids:
            s = id_to_section.get(sid)
            if s is None or s.is_dead:
                continue
            s.grow(opts.growth_rate, opts.time_step)
            s.update()
            if s.is_tip and not s.is_dead:
                logger.debug("TIP pos=%s len=%.2f age=%.2f", s.end, s.length, s.age)
        self._t_grow += (time.perf_counter() - t0)

        # ---------------------------------------------------------------------
        # Field precompute (if needed)
        # ---------------------------------------------------------------------
        t1 = time.perf_counter()
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
        self._t_fields += (time.perf_counter() - t1)

        # ---------------------------------------------------------------------
        # PASS 1: destructors
        # ---------------------------------------------------------------------
        t2 = time.perf_counter()
        r2 = getattr(opts, "neighbour_radius", 0.0) ** 2
        tips_snapshot = self._get_tips(mycel)
        tip_coords = {int(t.id): t.end.coords for t in tips_snapshot}
        for sid in ids:
            s = id_to_section.get(sid)
            if s is None or not s.is_tip or s.is_dead:
                continue
            if getattr(opts, "die_if_old", False) and s.age > opts.max_age:
                s.is_dead = True
                logger.debug("Tip died of age: age=%.2f > max_age=%.2f", s.age, opts.max_age)
                continue
            if s.length > opts.max_length:
                s.is_dead = True
                logger.debug("Tip died of length: len=%.2f > max_len=%.2f", s.length, opts.max_length)
                continue
            if need_density and s.field_aggregator:
                dens = density_map.get(int(s.id), s.field_aggregator.compute_field(s.end)[0])
                if dens > opts.density_threshold:
                    logger.debug("Density kill: %.3f > threshold %.3f", dens, opts.density_threshold)
                    s.is_dead = True
                    continue
            if need_nutrient and s.field_aggregator:
                nf = nutrient_map.get(int(s.id), s.field_aggregator.compute_field(s.end)[0])
                if nf < -abs(opts.nutrient_repulsion):
                    logger.debug("Repellent kill: nutrient_field=%.3f < -|repulsion|=%.3f",
                                 nf, abs(opts.nutrient_repulsion))
                    s.is_dead = True
                    continue
            if s.field_aggregator and r2 > 0.0:
                sx, sy, sz = tip_coords.get(int(s.id), s.end.coords)
                nearby = 0
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
        self._t_destruct += (time.perf_counter() - t2)

        # ---------------------------------------------------------------------
        # PASS 2: branching
        # ---------------------------------------------------------------------
        t3 = time.perf_counter()
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
        self._t_branch += (time.perf_counter() - t3)

        # ---------------------------------------------------------------------
        # End-of-step wrapper (unchanged)
        # ---------------------------------------------------------------------
        t4 = time.perf_counter()
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

        grid.update_from_mycel(mycel)
        if getattr(opts, "use_nutrient_field", False) and getattr(opts, "nutrient_repulsion", 0) > 0:
            if hasattr(mycel, "nutrient_kill_check"):
                mycel.nutrient_kill_check()
        mutator.apply(step, opts)
        checkpoints.maybe_save(mycel, step)
        stats.update(mycel)
        logger.debug(str(mycel))

        self._t_post += (time.perf_counter() - t4)
        self._steps += 1
