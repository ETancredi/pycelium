# parallel/engine.py

"""
Deterministic parallel step engine with lightweight timing instrumentation.

Step 1: Deterministic but SERIAL orientator.
- Enables `deterministic_orientator` path (per-tip RNG seeded by master_seed, step, tip.id)
- Keeps orientation compute fully serial (no batching, no parallel threads)
- Guarantees byte-for-byte equality with the original single-threaded model

Set env PYCELIUM_TIMINGS=1 to print a per-phase timing summary at shutdown.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging, os, time
import numpy as np

from core.options import Options
from core.section import Section

logger = logging.getLogger("pycelium.parallel.engine")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _take_snapshot_ids(sections: List[Section]) -> List[int]:
    """Stable deterministic ordering of section IDs."""
    return [int(s.id) for s in sections]


# ---------------------------------------------------------------------------
# ParallelStepEngine
# ---------------------------------------------------------------------------

class ParallelStepEngine:
    def __init__(self, apply_order: str = "id", workers: int = 0):
        self.apply_order = apply_order
        self.workers = int(workers)
        self._pool = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

        # timing accumulators (seconds)
        self._t_orient = 0.0
        self._t_grow = 0.0
        self._t_fields = 0.0
        self._t_destruct = 0.0
        self._t_branch = 0.0
        self._t_post = 0.0
        self._steps = 0

    # -----------------------------------------------------------------------
    # Shutdown / reporting
    # -----------------------------------------------------------------------
    def shutdown(self):
        """Cleanly shut down the persistent pool (call after the run)."""
        if os.getenv("PYCELIUM_TIMINGS", "").lower() in ("1", "true", "yes"):
            total = (
                self._t_orient + self._t_grow + self._t_fields +
                self._t_destruct + self._t_branch + self._t_post
            )
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

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _get_tips(self, mycel):
        """Return list of active (non-dead) tip sections."""
        return [s for s in mycel.sections if s.is_tip and not s.is_dead]

    # -----------------------------------------------------------------------
    # Orientation computation  (Step 1 version: deterministic + serial)
    # -----------------------------------------------------------------------
    def _compute_orientations(
        self,
        tips: List[Section],
        orientator,
        opts: Options,
        step: int,
        master_seed: Optional[int],
    ):
        """Compute orientations deterministically (serial per tip)."""
        if not tips:
            return

        ordered = list(tips)  # stable order

        # STEP 1: deterministic orientator enabled, but still serial
        if bool(getattr(opts, "deterministic_orientator", False)):
            for tip in ordered:
                tip.orientation = orientator.compute_deterministic(
                    tip, step=step, master_seed=master_seed
                )
            return

        # Legacy non-deterministic path (global RNG)
        for tip in ordered:
            tip.orientation = orientator.compute(tip)

    # -----------------------------------------------------------------------
    # Main step
    # -----------------------------------------------------------------------
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

        # ORIENTATIONS (deterministic, serial in Step 1)
        t_or = time.perf_counter()
        self._compute_orientations(self._get_tips(mycel), orientator, opts, step, master_seed)
        self._t_orient += (time.perf_counter() - t_or)

        # Snapshot ids and id→section map
        ids = _take_snapshot_ids(mycel.sections)
        id_to_section = {int(s.id): s for s in mycel.sections}
        logger.debug("STEP START: t=%.2f | total_sections=%d", mycel.time, len(mycel.sections))

        # PASS 0: grow / update
        t0 = time.perf_counter()
        for sid in ids:
            s = id_to_section.get(sid)
            if s is None or s.is_dead:
                continue
            s.grow(opts.growth_rate, opts.time_step)
            s.update()
        self._t_grow += (time.perf_counter() - t0)

        # Field precompute (density/nutrient)
        t1 = time.perf_counter()
        need_density  = bool(getattr(opts, "die_if_too_dense", False))
        need_nutrient = bool(getattr(opts, "use_nutrient_field", False)
                             and getattr(opts, "nutrient_repulsion", 0) != 0)
        density_map: Dict[int, float] = {}
        nutrient_map: Dict[int, float] = {}
        current_tips = self._get_tips(mycel)
        if (need_density or need_nutrient) and current_tips:
            tip_sids = [int(t.id) for t in current_tips]
            def _field_at_end(t: Section) -> float:
                agg = t.field_aggregator
                return agg.compute_field(t.end)[0] if agg else 0.0
            values = [_field_at_end(t) for t in current_tips]
            for sid, val in zip(tip_sids, values):
                if need_density:
                    density_map[sid] = val
                if need_nutrient:
                    nutrient_map[sid] = val
        self._t_fields += (time.perf_counter() - t1)

        # PASS 1: destructors
        t2 = time.perf_counter()
        r2 = getattr(opts, "neighbour_radius", 0.0) ** 2
        tips_snapshot = self._get_tips(mycel)
        tip_coords = {int(t.id): t.end.coords for t in tips_snapshot}
        for sid in ids:
            s = id_to_section.get(sid)
            if s is None or not s.is_tip or s.is_dead:
                continue
            # Age / length kills
            if getattr(opts, "die_if_old", False) and s.age > opts.max_age:
                s.is_dead = True
                continue
            if s.length > opts.max_length:
                s.is_dead = True
                continue
            # Density kill
            if need_density and s.field_aggregator:
                dens = density_map.get(int(s.id), s.field_aggregator.compute_field(s.end)[0])
                if dens > opts.density_threshold:
                    s.is_dead = True
                    continue
            # Nutrient kill
            if need_nutrient and s.field_aggregator:
                nf = nutrient_map.get(int(s.id), s.field_aggregator.compute_field(s.end)[0])
                if nf < -abs(opts.nutrient_repulsion):
                    s.is_dead = True
                    continue
            # Isolation check
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
                    s.is_dead = True
                    continue
        self._t_destruct += (time.perf_counter() - t2)

        # PASS 2: branching
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
                    new_sections.append(child)
        if new_sections:
            mycel.sections.extend(new_sections)
        self._t_branch += (time.perf_counter() - t3)

        # PASS 3: wrap-up / book-keeping
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

        # Tip pruning (if configured)
        if hasattr(opts, "max_supported_tips") and opts.max_supported_tips > 0:
            active_tips = self._get_tips(mycel)
            if len(active_tips) > opts.max_supported_tips:
                excess = len(active_tips) - opts.max_supported_tips
                to_prune = np.random.choice(active_tips, size=excess, replace=False)
                for tip in to_prune:
                    tip.is_dead = True

        # Biomass / series update
        tip_data = [(tip.end.coords[0], tip.end.coords[1], tip.end.coords[2])
                    for tip in self._get_tips(mycel)]
        mycel.step_history.append((mycel.time, tip_data))
        total_biomass = sum(sec.length for sec in mycel.sections if not sec.is_dead)
        mycel.biomass_history.append(total_biomass)

        grid.update_from_mycel(mycel)
        if getattr(opts, "use_nutrient_field", False) and getattr(opts, "nutrient_repulsion", 0) > 0:
            if hasattr(mycel, "nutrient_kill_check"):
                mycel.nutrient_kill_check()
        mutator.apply(step, opts)
        checkpoints.maybe_save(mycel, step)
        stats.update(mycel)

        self._t_post += (time.perf_counter() - t4)
        self._steps += 1
