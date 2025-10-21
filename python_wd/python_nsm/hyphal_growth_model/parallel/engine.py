# parallel/engine.py
"""
Deterministic parallel step engine with lightweight timing instrumentation.

Now supports:
- Step 1: deterministic (serial) orientator
- Step 2: deterministic *batched* (serial) orientator
- Step 3: parallel orientator with chunking:
    - backend="thread": threads (useful if heavy NumPy kernels release the GIL)
    - backend="process": true multicore; we precompute fields in main proc and
      farm out the *combination math* to processes (so Aggregator needn't be pickled)

Env: PYCELIUM_TIMINGS=1 prints a per-phase summary at shutdown.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import logging, os, time
import numpy as np

from core.options import Options
from core.section import Section
from core.point import MPoint

logger = logging.getLogger("pycelium.parallel.engine")


# --------- helpers -----------------------------------------------------------

def _take_snapshot_ids(sections: List[Section]) -> List[int]:
    return [int(s.id) for s in sections]

def _chunks(seq, size: int):
    for i in range(0, len(seq), max(1, size)):
        yield seq[i:i+size]

def _mps_to_array(pts: List[MPoint]) -> np.ndarray:
    if not pts:
        return np.zeros((0,3), dtype=float)
    return np.array([p.coords for p in pts], dtype=float)

def _array_to_mps(arr: np.ndarray) -> List[MPoint]:
    return [MPoint(float(x), float(y), float(z)) for (x,y,z) in arr]


# --------- process worker: pure numpy combination (picklable) ---------------

def _combine_numpy_worker(args: Tuple[
    np.ndarray,  # base_orients Nx3
    np.ndarray,  # grads Nx3 (NaN where None)
    np.ndarray,  # curvs Nx1 (NaN if unused)
    np.ndarray,  # density_grads Nx3
    np.ndarray,  # aniso_dirs Nx3
    np.ndarray,  # rand_vecs Nx3
    np.ndarray,  # nutrient_vecs Nx3
    Dict[str, float],  # scalar opts used in math
    np.ndarray,  # prev_orients Nx3
]) -> np.ndarray:
    (base_orients, grads, curvs, dens, aniso, randv, nutr, scalars, prev_orients) = args

    # unpack scalars
    autotropism               = scalars["autotropism"]
    align_boost               = scalars["field_alignment_boost"]
    curv_infl                 = scalars["field_curvature_influence"]
    die_if_too_dense          = bool(scalars["die_if_too_dense"])
    gravitropism              = scalars["gravitropism"]
    gravi_start               = scalars["gravi_angle_start"]
    gravi_end                 = scalars["gravi_angle_end"]
    aniso_enabled             = bool(scalars["anisotropy_enabled"])
    aniso_strength            = scalars["anisotropy_strength"]
    random_walk               = scalars["random_walk"]
    memory_blend              = scalars["direction_memory_blend"]

    ori = base_orients.copy()

    # autotropism
    have_grad = ~np.isnan(grads).any(axis=1)
    ori[have_grad] += autotropism * grads[have_grad]

    # alignment boost
    if align_boost > 0.0:
        gu = grads.copy()
        # normalise safe
        gnorm = np.linalg.norm(gu, axis=1, keepdims=True)
        mask = (gnorm[:,0] > 0) & have_grad
        gu[mask] = gu[mask] / gnorm[mask]
        # dot with ori (before adding boost)
        dot = np.einsum("ij,ij->i", ori, gu, where=mask[:,None])
        pos = (dot > 0) & mask
        ori[pos] += (align_boost * dot[pos])[:,None] * gu[pos]

    # curvature
    if curv_infl > 0.0:
        # reuse gu (unit grad where valid)
        curv_mask = (~np.isnan(curvs[:,0])) & have_grad
        ori[curv_mask] += (curv_infl * curvs[curv_mask]) * gu[curv_mask]

    # density avoidance
    if die_if_too_dense:
        ori -= dens

    # gravitropism
    if gravitropism > 0.0:
        z = 0.0 + 0*ori[:,0]  # placeholder (we don't have z per tip here)
        # In the engine we pass the real z as a 1D array via curvs[:,0] hack is messy;
        # Instead we piggyback it in 'nutr' extra column if needed. Simpler:
        # We won't compute z-based ramp here; engine precomputed strength per tip in 'randv'[:,0]? No.
        # To keep the worker pure & simple, we let engine pass a ready-made grav_vec Nx3:
        # -> we will *not* do gravitropism here; engine adds it into aniso (already scaled).
        pass  # no-op; engine pre-adds the gravity contribution to 'aniso'.

    # anisotropy + gravity (engine pre-summed into 'aniso')
    if aniso_enabled or True:
        ori += aniso_strength * aniso

    # random walk
    if random_walk > 0.0:
        # normalise rows
        rn = np.linalg.norm(randv, axis=1, keepdims=True)
        maskr = rn[:,0] > 0
        rv = randv.copy()
        rv[maskr] = rv[maskr] / rn[maskr]
        ori += random_walk * rv

    # nutrient (engine passes the final nutrient vector already signed/scaled)
    ori += nutr

    # memory blend
    if memory_blend > 0.0:
        ori = (memory_blend * prev_orients) + ((1.0 - memory_blend) * ori)

    # normalise output
    nrm = np.linalg.norm(ori, axis=1, keepdims=True)
    nz = nrm[:,0] > 0
    ori[nz] = ori[nz] / nrm[nz]
    return ori


# --------- Engine ------------------------------------------------------------

class ParallelStepEngine:
    def __init__(self, apply_order: str = "id", workers: int = 0):
        self.apply_order = apply_order
        self.workers = int(workers)

        # pools created lazily when used:
        self._tpool: Optional[ThreadPoolExecutor] = None
        self._ppool: Optional[ProcessPoolExecutor] = None

        # timing accumulators (seconds)
        self._t_orient = 0.0
        self._t_grow = 0.0
        self._t_fields = 0.0
        self._t_destruct = 0.0
        self._t_branch = 0.0
        self._t_post = 0.0
        self._steps = 0

    # -- pool accessors -------------------------------------------------------
    def _thread_pool(self):
        if self._tpool is None and self.workers > 1:
            self._tpool = ThreadPoolExecutor(max_workers=self.workers)
        return self._tpool

    def _proc_pool(self):
        if self._ppool is None and self.workers > 1:
            # maCOS uses 'spawn' -> safe for pickling
            self._ppool = ProcessPoolExecutor(max_workers=self.workers)
        return self._ppool

    def shutdown(self):
        """Cleanly shut down persistent pools."""
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

        if self._tpool is not None:
            self._tpool.shutdown(wait=True)
            self._tpool = None
        if self._ppool is not None:
            self._ppool.shutdown(wait=True)
            self._ppool = None

    # helpers

    def _get_tips(self, mycel):
        return [s for s in mycel.sections if s.is_tip and not s.is_dead]

    # -- orientator driver ----------------------------------------------------

    def _compute_orientations(
        self,
        tips: List[Section],
        orientator,
        opts: Options,
        step: int,
        master_seed: Optional[int],
    ):
        if not tips:
            return

        ordered = list(tips)  # stable

        det = bool(getattr(opts, "deterministic_orientator", True))
        batched = bool(getattr(opts, "orientator_batched", True))
        allow_parallel = bool(getattr(opts, "parallelise_orientator", False))
        backend = (getattr(opts, "orientator_backend", "thread") or "thread").lower()
        chunk_size = int(getattr(opts, "orientator_chunk_size", 1024))

        # --- Step 1: deterministic serial (no batching)
        if det and not batched:
            for tip in ordered:
                tip.orientation = orientator.compute_deterministic(tip, step=step, master_seed=master_seed)
            return

        # --- Step 2: deterministic *batched* serial
        if det and batched and not allow_parallel:
            res = orientator.compute_many_deterministic_batched(ordered, step=step, master_seed=master_seed)
            for tip, ori in zip(ordered, res):
                tip.orientation = ori
            return

        # --- Step 3: parallel batched ---
        if det and batched and allow_parallel:
            if backend == "thread":
                # Threads: invoke batched method per chunk. Works well if aggregator’s
                # batched methods drop the GIL (NumPy/C code inside).
                pool = self._thread_pool()
                if pool is None:
                    # fallback to serial batched
                    res = orientator.compute_many_deterministic_batched(ordered, step=step, master_seed=master_seed)
                    for tip, ori in zip(ordered, res):
                        tip.orientation = ori
                    return

                chunks = list(_chunks(ordered, chunk_size))
                def _run_chunk(chunk):
                    return orientator.compute_many_deterministic_batched(chunk, step=step, master_seed=master_seed)
                results = list(pool.map(_run_chunk, chunks))
                for chunk, res in zip(chunks, results):
                    for tip, ori in zip(chunk, res):
                        tip.orientation = ori
                return

            elif backend == "process":
                # Processes: we must NOT pickle the orientator/aggregator.
                # Strategy: precompute everything in main proc, then farm out
                # the *pure combination math* to processes.
                pool = self._proc_pool()
                if pool is None:
                    # fallback to serial batched
                    res = orientator.compute_many_deterministic_batched(ordered, step=step, master_seed=master_seed)
                    for tip, ori in zip(ordered, res):
                        tip.orientation = ori
                    return

                # --- PRECOMPUTE in main process ---
                n = len(ordered)
                base_orients = _mps_to_array([t.orientation for t in ordered])
                prev_orients = base_orients.copy()

                # gradients (batched)
                grads = np.full((n,3), np.nan, dtype=float)
                if orientator.aggregator is not None:
                    pts = [t.end for t in ordered]
                    _, g_list = orientator.aggregator.compute_field_many(pts)
                    g_arr = _mps_to_array([g if g is not None else MPoint(0,0,0) for g in g_list])
                    # mark invalid grads as NaN rows
                    valid = np.array([g is not None for g in g_list], dtype=bool)
                    grads[valid] = g_arr[valid]
                    # (invalid rows remain NaN)

                # curvature
                curvs = np.full((n,1), np.nan, dtype=float)
                if getattr(opts, "field_curvature_influence", 0.0) > 0.0 and orientator.aggregator is not None:
                    pts = [t.end for t in ordered]
                    if hasattr(orientator.aggregator, "compute_field_curvature_many"):
                        cvals = orientator.aggregator.compute_field_curvature_many(pts)
                    else:
                        cvals = [orientator.aggregator.compute_field_curvature(p) for p in pts]
                    curvs[:,0] = np.asarray(cvals, dtype=float)

                # density gradients (per tip, cheap)
                if getattr(opts, "die_if_too_dense", False) and orientator.density_grid:
                    dens = _mps_to_array([orientator.density_grid.get_gradient_at(t.end) for t in ordered])
                else:
                    dens = np.zeros((n,3), dtype=float)

                # anisotropy direction + gravity (pre-summed into aniso)
                if getattr(opts, "anisotropy_enabled", False):
                    aniso_dirs = _mps_to_array([
                        (orientator.anisotropy_grid.get_direction_at(t.end)
                         if orientator.anisotropy_grid else
                         MPoint(*opts.anisotropy_vector).normalise())
                        for t in ordered
                    ])
                else:
                    aniso_dirs = np.zeros((n,3), dtype=float)

                # gravity vector per tip (we add it into aniso_dirs scaled by strength)
                if getattr(opts, "gravitropism", 0.0) > 0.0:
                    gvec = np.tile(np.array([0.0, -1.0, 0.0], dtype=float), (n,1))
                    z = np.array([t.end.coords[2] for t in ordered], dtype=float)
                    start, end, gstr = float(opts.gravi_angle_start), float(opts.gravi_angle_end), float(opts.gravitropism)
                    strength = np.where(z < start, 0.0,
                                np.where(z > end, gstr,
                                         gstr * ((z - start) / max(1e-12, (end - start)))))
                    aniso_dirs = aniso_dirs + (strength[:,None] * gvec)

                # nutrient resultant (per tip, signed/scaled like serial)
                if (getattr(opts, "nutrient_radius", 0.0) > 0.0) and (orientator.nutrient_sources):
                    nutr = np.zeros((n,3), dtype=float)
                    R = float(opts.nutrient_radius)
                    attr = float(getattr(opts, "nutrient_attraction", 0.0))
                    rep  = float(getattr(opts, "nutrient_repulsion", 0.0))
                    srcs = np.array([p.coords for p in orientator.nutrient_sources], dtype=float)  # Sx3
                    ends = np.array([t.end.coords for t in ordered], dtype=float)                  # Nx3
                    for si in range(srcs.shape[0]):
                        delta = srcs[si][None,:] - ends  # Nx3
                        dist = np.linalg.norm(delta, axis=1)
                        mask = dist < R
                        if not np.any(mask):
                            continue
                        unit = delta[mask] / np.maximum(dist[mask,None], 1e-12)
                        infl = 1.0 - (dist[mask] / R)
                        if attr > 0:
                            nutr[mask] += (attr * infl)[:,None] * unit
                        if rep > 0:
                            nutr[mask] -= (rep  * infl)[:,None] * unit
                else:
                    nutr = np.zeros((n,3), dtype=float)

                # RNG (per-tip deterministic)
                randv = np.zeros((n,3), dtype=float)
                if getattr(opts, "random_walk", 0.0) > 0.0:
                    from numpy.random import Generator, PCG64
                    for i, tip in enumerate(ordered):
                        seed64 = int(orientator._combine_seed(master_seed, step, int(tip.id)))  # type: ignore[attr-defined]
                        g = Generator(PCG64(seed64))
                        randv[i,:] = g.normal(0.0, 1.0, 3)

                # scalars used inside worker
                scalars = dict(
                    autotropism=float(getattr(opts, "autotropism", 0.0)),
                    field_alignment_boost=float(getattr(opts, "field_alignment_boost", 0.0)),
                    field_curvature_influence=float(getattr(opts, "field_curvature_influence", 0.0)),
                    die_if_too_dense=float(1.0 if getattr(opts, "die_if_too_dense", False) else 0.0),
                    gravitropism=float(getattr(opts, "gravitropism", 0.0)),
                    gravi_angle_start=float(getattr(opts, "gravi_angle_start", 0.0)),
                    gravi_angle_end=float(getattr(opts, "gravi_angle_end", 1.0)),
                    anisotropy_enabled=float(1.0 if getattr(opts, "anisotropy_enabled", False) else 0.0),
                    anisotropy_strength=float(getattr(opts, "anisotropy_strength", 0.0)),
                    random_walk=float(getattr(opts, "random_walk", 0.0)),
                    direction_memory_blend=float(getattr(opts, "direction_memory_blend", 0.0)),
                )

                # chunk the arrays
                idx_chunks = list(_chunks(list(range(n)), chunk_size))
                arg_chunks = []
                for idxs in idx_chunks:
                    sl = np.array(idxs, dtype=int)
                    arg_chunks.append((
                        base_orients[sl],
                        grads[sl],
                        curvs[sl],
                        dens[sl],
                        aniso_dirs[sl],
                        randv[sl],
                        nutr[sl],
                        scalars,
                        prev_orients[sl],
                    ))

                # map to processes
                results = list(pool.map(_combine_numpy_worker, arg_chunks))

                # stitch back
                out = np.vstack(results) if results else np.zeros((0,3), dtype=float)
                # maintain original order (idx_chunks already contiguous)
                k = 0
                for tip in ordered:
                    tip.orientation = MPoint(*out[k,:])
                    k += 1
                return

            else:
                logger.warning("Unknown orientator_backend=%r; falling back to serial batched.", backend)
                res = orientator.compute_many_deterministic_batched(ordered, step=step, master_seed=master_seed)
                for tip, ori in zip(ordered, res):
                    tip.orientation = ori
                return

        # --- legacy non-deterministic path (serial) ---
        for tip in ordered:
            tip.orientation = orientator.compute(tip)

    # main step
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

        # ORIENTATIONS
        t_or = time.perf_counter()
        self._compute_orientations(self._get_tips(mycel), orientator, opts, step, master_seed)
        self._t_orient += (time.perf_counter() - t_or)

        # Snapshot ids and id->section map
        ids = _take_snapshot_ids(mycel.sections)
        id_to_section = {int(s.id): s for s in mycel.sections}

        logger.debug("STEP START: t=%.2f | total_sections=%d", mycel.time, len(mycel.sections))

        # PASS 0: grow/update
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

        # Field precompute (if needed) (unchanged)
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
            pool = self._thread_pool()
            if pool is not None:
                values = list(pool.map(_field_at_end, current_tips))
            else:
                values = [_field_at_end(t) for t in current_tips]
            for sid, val in zip(tip_sids, values):
                if need_density:
                    density_map[sid] = val
                if need_nutrient:
                    nutrient_map[sid] = val
        self._t_fields += (time.perf_counter() - t1)

        # PASS 1: destructors (unchanged)
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

        # PASS 2: branching (unchanged)
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

        # End-of-step wrapper (unchanged)
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
