# tropisms/orientator.py

from __future__ import annotations
from typing import List, Optional
from core.point import MPoint
from core.section import Section
from core.options import Options
from compute.field_aggregator import FieldAggregator
from vis.anisotropy_grid import AnisotropyGrid
import numpy as np
from numpy.random import Generator, PCG64
import logging

logger = logging.getLogger("pycelium")


# ---- stable seeding helpers (masked to avoid overflow warnings) ----
def _stable_uint64(x: int) -> np.uint64:
    """
    SplitMix64-style scrambler to derive a stable 64-bit value from an int.
    Deterministic across platforms/threads. Mask after each step to avoid overflow warnings.
    """
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
    z = (np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)) & mask
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    z = (z ^ (z >> np.uint64(31))) & mask
    return z


def _combine_seed(master_seed: Optional[int], step: int, tip_id: int) -> np.uint64:
    """
    Combine (master_seed, step, tip_id) into a deterministic 64-bit seed.
    """
    base = int(master_seed) if master_seed is not None else 0
    z = _stable_uint64(base)
    z = _stable_uint64(int(z) ^ (step & 0xFFFFFFFF))
    z = _stable_uint64(int(z) ^ (tip_id & 0xFFFFFFFF))
    return z


class Orientator:
    """
    Combines multiple tropism influences to compute new growth orientation.

    Step 1 baseline: deterministic, thread-safe, *serial* path via `compute_deterministic`.
    Parallel batching is *not* enabled here; behaviour matches the legacy serial model.
    """
    def __init__(self, options: Options):
        self.options = options
        self.aggregator: FieldAggregator | None = None
        self.density_grid = None
        self.anisotropy_grid: AnisotropyGrid | None = None
        self.nutrient_sources: List[MPoint] = []

    def set_field_source(self, aggregator: FieldAggregator):
        self.aggregator = aggregator

    def set_density_grid(self, grid):
        self.density_grid = grid

    def set_anisotropy_grid(self, grid: AnisotropyGrid):
        self.anisotropy_grid = grid

    def set_nutrient_sources(self, points: List[MPoint]):
        self.nutrient_sources = points

    # -------------------- deterministic serial entrypoint --------------------
    def compute_deterministic(self, section: Section, *, step: int, master_seed: Optional[int]) -> MPoint:
        """
        Deterministic, thread-safe orientation compute:
        - Creates a local RNG seeded by (master_seed, step, section.id).
        - No global RNG or shared-state mutation.
        - Mirrors the legacy `compute()` math.
        """
        seed64 = int(_combine_seed(master_seed, step, int(section.id)))
        rng = Generator(PCG64(seed64))
        return self._compute_with_rng(section, rng)

    # Helper used by the deterministic path (serial)
    def _compute_with_rng(self, section: Section, rng: Generator) -> MPoint:
        opts = self.options
        orientation = section.orientation.copy()

        # Autotropism & Field Alignment
        grad = None
        if self.aggregator:
            _, grad = self.aggregator.compute_field(section.end)  # scalar, gradient
            if grad is not None:
                orientation.add(grad.scale(float(opts.autotropism)))

                # Alignment boost
                if opts.field_alignment_boost > 0:
                    grad_unit = grad.copy().normalise()
                    dot = orientation.dot(grad_unit)
                    if dot > 0:
                        orientation.add(grad_unit.scale(dot * float(opts.field_alignment_boost)))

                # Curvature
                if opts.field_curvature_influence > 0:
                    curvature = self.aggregator.compute_field_curvature(section.end)
                    direction = grad.copy().normalise()
                    orientation.add(direction.scale(curvature * float(opts.field_curvature_influence)))

        # Density-based avoidance
        if opts.die_if_too_dense and self.density_grid:
            density_grad = self.density_grid.get_gradient_at(section.end)
            orientation.subtract(density_grad)

        # Gravitropism
        if opts.gravitropism > 0:
            z = section.end.coords[2]
            if z < opts.gravi_angle_start:
                strength = 0.0
            elif z > opts.gravi_angle_end:
                strength = float(opts.gravitropism)
            else:
                t = (z - opts.gravi_angle_start) / (opts.gravi_angle_end - opts.gravi_angle_start)
                strength = float(t * opts.gravitropism)
            orientation.add(MPoint(0, -1, 0).scale(strength))

        # Nutrient fields
        for nutrient in self.nutrient_sources:
            delta = nutrient.copy().subtract(section.end)
            dist = delta.magnitude()
            if dist < opts.nutrient_radius:
                influence = 1.0 - (dist / opts.nutrient_radius)
                if opts.nutrient_attraction > 0:
                    orientation.add(delta.copy().normalise().scale(float(opts.nutrient_attraction) * influence))
                if opts.nutrient_repulsion > 0:
                    orientation.subtract(delta.copy().normalise().scale(float(opts.nutrient_repulsion) * influence))

        # Anisotropy
        if opts.anisotropy_enabled:
            dir_vec = (
                self.anisotropy_grid.get_direction_at(section.end)
                if self.anisotropy_grid else
                MPoint(*opts.anisotropy_vector).normalise()
            )
            orientation.add(dir_vec.scale(float(opts.anisotropy_strength)))

        # Random walk (use provided RNG)
        if opts.random_walk > 0:
            rand = rng.normal(0.0, 1.0, 3)
            orientation.add(MPoint(*rand).normalise().scale(float(opts.random_walk)))

        # Directional memory blending
        if opts.direction_memory_blend > 0 and section.orientation:
            blend = float(opts.direction_memory_blend)
            orientation = section.orientation.copy().scale(blend).add(orientation.scale(1.0 - blend)).normalise()

        return orientation.normalise()

    # -------------------- legacy serial API (unchanged) ----------------------
    def compute(self, section: Section) -> MPoint:
        """
        Original API: uses global RNG (np.random). Not thread-safe deterministic.
        Kept for backwards compatibility / single-threaded mode.
        """
        orientation = section.orientation.copy()

        grad = None
        if self.aggregator:
            _, grad = self.aggregator.compute_field(section.end)
            if grad is not None:
                orientation.add(grad.scale(self.options.autotropism))

                if self.options.field_alignment_boost > 0:
                    grad_unit = grad.copy().normalise()
                    dot = orientation.dot(grad_unit)
                    if dot > 0:
                        orientation.add(grad_unit.scale(dot * self.options.field_alignment_boost))

                if self.options.field_curvature_influence > 0:
                    curvature = self.aggregator.compute_field_curvature(section.end)
                    direction = grad.copy().normalise()
                    orientation.add(direction.scale(curvature * self.options.field_curvature_influence))

        if self.options.die_if_too_dense and self.density_grid:
            orientation.subtract(self.density_grid.get_gradient_at(section.end))

        if self.options.gravitropism > 0:
            z = section.end.coords[2]
            if z < self.options.gravi_angle_start:
                strength = 0.0
            elif z > self.options.gravi_angle_end:
                strength = float(self.options.gravitropism)
            else:
                t = (z - self.options.gravi_angle_start) / (self.options.gravi_angle_end - self.options.gravi_angle_start)
                strength = float(t * self.options.gravitropism)
            orientation.add(MPoint(0, -1, 0).scale(strength))

        for nutrient in self.nutrient_sources:
            delta = nutrient.copy().subtract(section.end)
            dist = delta.magnitude()
            if dist < self.options.nutrient_radius:
                influence = 1.0 - (dist / self.options.nutrient_radius)
                if self.options.nutrient_attraction > 0:
                    orientation.add(delta.copy().normalise().scale(float(self.options.nutrient_attraction) * influence))
                if self.options.nutrient_repulsion > 0:
                    orientation.subtract(delta.copy().normalise().scale(float(self.options.nutrient_repulsion) * influence))

        if self.options.anisotropy_enabled:
            dir_vec = (
                self.anisotropy_grid.get_direction_at(section.end)
                if self.anisotropy_grid else
                MPoint(*self.options.anisotropy_vector).normalise()
            )
            orientation.add(dir_vec.scale(float(self.options.anisotropy_strength)))

        if self.options.random_walk > 0:
            rand = np.random.normal(0, 1, 3)
            orientation.add(MPoint(*rand).normalise().scale(float(self.options.random_walk)))

        if self.options.direction_memory_blend > 0 and section.orientation:
            blend = float(self.options.direction_memory_blend)
            orientation = section.orientation.copy().scale(blend).add(orientation.scale(1.0 - blend)).normalise()

        return orientation.normalise()

    # -------------------- batch helpers (NOT used in Step 1) -----------------
    # Kept for later steps; currently unused so behaviour is unchanged.
    def compute_many(self, tips: List[Section]) -> List[MPoint]:
        n = len(tips)
        if n == 0:
            return []
        out_orients = [t.orientation.copy() for t in tips]
        grad_list = [None] * n
        if self.aggregator is not None:
            points = [t.end for t in tips]
            _, grads = self.aggregator.compute_field_many(points)
            grad_list = grads
        opts = self.options
        for i, tip in enumerate(tips):
            ori = out_orients[i]
            grad = grad_list[i]
            if grad is not None:
                ori.add(grad.copy().scale(float(opts.autotropism)))
                if opts.field_alignment_boost > 0:
                    grad_unit = grad.copy().normalise()
                    dot = ori.dot(grad_unit)
                    if dot > 0:
                        ori.add(grad_unit.scale(dot * float(opts.field_alignment_boost)))
                if opts.field_curvature_influence > 0 and self.aggregator is not None:
                    curvature = self.aggregator.compute_field_curvature(tip.end)
                    direction = grad.copy().normalise()
                    ori.add(direction.scale(curvature * float(opts.field_curvature_influence)))
            if opts.die_if_too_dense and self.density_grid:
                ori.subtract(self.density_grid.get_gradient_at(tip.end))
            if opts.gravitropism > 0:
                z = tip.end.coords[2]
                if z < opts.gravi_angle_start: strength = 0.0
                elif z > opts.gravi_angle_end: strength = float(opts.gravitropism)
                else:
                    t = (z - opts.gravi_angle_start) / (opts.gravi_angle_end - opts.gravi_angle_start)
                    strength = float(t * opts.gravitropism)
                ori.add(MPoint(0, -1, 0).scale(strength))
            for nutrient in self.nutrient_sources:
                delta = nutrient.copy().subtract(tip.end)
                dist = delta.magnitude()
                if dist < opts.nutrient_radius:
                    influence = 1.0 - (dist / opts.nutrient_radius)
                    if opts.nutrient_attraction > 0:
                        ori.add(delta.copy().normalise().scale(float(opts.nutrient_attraction) * influence))
                    if opts.nutrient_repulsion > 0:
                        ori.subtract(delta.copy().normalise().scale(float(opts.nutrient_repulsion) * influence))
            if opts.anisotropy_enabled:
                dir_vec = self.anisotropy_grid.get_direction_at(tip.end) if self.anisotropy_grid else MPoint(*opts.anisotropy_vector).normalise()
                ori.add(dir_vec.scale(float(opts.anisotropy_strength)))
            if opts.random_walk > 0:
                rand = np.random.normal(0, 1, 3)
                ori.add(MPoint(*rand).normalise().scale(float(opts.random_walk)))
            if opts.direction_memory_blend > 0 and tip.orientation:
                blend = float(opts.direction_memory_blend)
                ori = tip.orientation.copy().scale(blend).add(ori.scale(1.0 - blend)).normalise()
            out_orients[i] = ori.normalise()
        return out_orients

    def compute_many_deterministic(
        self,
        tips: List[Section],
        *,
        step: int,
        master_seed: Optional[int],
    ) -> List[MPoint]:
        """
        Deterministic, chunk-friendly orientation compute for a list of tips.
        NOTE: We intentionally keep per-tip aggregator calls (no field batching) so
        behaviour stays byte-for-byte equal to the serial deterministic path.
        """
        n = len(tips)
        if n == 0:
            return []

        opts = self.options
        out_orients = [t.orientation.copy() for t in tips]

        # RNG batch for random-walk (still per-tip seeded)
        rand_vectors = None
        if opts.random_walk > 0:
            rand_vectors = np.empty((n, 3), dtype=float)
            for i, tip in enumerate(tips):
                seed64 = int(_combine_seed(master_seed, step, int(tip.id)))
                g = Generator(PCG64(seed64))
                rand_vectors[i, :] = g.normal(0.0, 1.0, 3)

        for i, tip in enumerate(tips):
            ori = out_orients[i]

            grad = None
            if self.aggregator:
                _, grad = self.aggregator.compute_field(tip.end)
                if grad is not None:
                    ori.add(grad.copy().scale(float(opts.autotropism)))
                    if opts.field_alignment_boost > 0:
                        grad_unit = grad.copy().normalise()
                        dot = ori.dot(grad_unit)
                        if dot > 0:
                            ori.add(grad_unit.scale(dot * float(opts.field_alignment_boost)))
                    if opts.field_curvature_influence > 0:
                        curvature = self.aggregator.compute_field_curvature(tip.end)
                        direction = grad.copy().normalise()
                        ori.add(direction.scale(curvature * float(opts.field_curvature_influence)))

            if opts.die_if_too_dense and self.density_grid:
                ori.subtract(self.density_grid.get_gradient_at(tip.end))

            if opts.gravitropism > 0:
                z = tip.end.coords[2]
                if z < opts.gravi_angle_start:
                    strength = 0.0
                elif z > opts.gravi_angle_end:
                    strength = float(opts.gravitropism)
                else:
                    t = (z - opts.gravi_angle_start) / (opts.gravi_angle_end - opts.gravi_angle_start)
                    strength = float(t * opts.gravitropism)
                ori.add(MPoint(0, -1, 0).scale(strength))

            for nutrient in self.nutrient_sources:
                delta = nutrient.copy().subtract(tip.end)
                dist = delta.magnitude()
                if dist < opts.nutrient_radius:
                    influence = 1.0 - (dist / opts.nutrient_radius)
                    if opts.nutrient_attraction > 0:
                        ori.add(delta.copy().normalise().scale(float(opts.nutrient_attraction) * influence))
                    if opts.nutrient_repulsion > 0:
                        ori.subtract(delta.copy().normalise().scale(float(opts.nutrient_repulsion) * influence))

            if opts.anisotropy_enabled:
                dir_vec = self.anisotropy_grid.get_direction_at(tip.end) if self.anisotropy_grid else MPoint(*opts.anisotropy_vector).normalise()
                ori.add(dir_vec.scale(float(opts.anisotropy_strength)))

            if opts.random_walk > 0 and rand_vectors is not None:
                rw = rand_vectors[i, :]
                ori.add(MPoint(*rw).normalise().scale(float(opts.random_walk)))

            if opts.direction_memory_blend > 0 and tip.orientation:
                blend = float(opts.direction_memory_blend)
                ori = tip.orientation.copy().scale(blend).add(ori.scale(1.0 - blend)).normalise()

            out_orients[i] = ori.normalise()

        return out_orients

    def compute_many_deterministic_batched(
        self,
        tips: List[Section],
        *,
        step: int,
        master_seed: Optional[int],
    ) -> List[MPoint]:
        """
        Deterministic, *batched* orientation compute:
        - Batch field gradients via aggregator.compute_field_many (if present)
        - Batch curvature via aggregator.compute_field_curvature_many (if present; else per-tip)
        - Deterministic per-tip RNG (PCG64(master_seed, step, tip.id))
        Preserves serial behaviour exactly.
        """
        n = len(tips)
        if n == 0:
            return []

        opts = self.options
        out_orients = [t.orientation.copy() for t in tips]

        # --- gradients (batched) ---
        grad_list: List[Optional[MPoint]] = [None] * n
        if self.aggregator is not None:
            points = [t.end for t in tips]
            _, grads = self.aggregator.compute_field_many(points)
            grad_list = grads

        # --- curvature (batched if available, else per-tip) ---
        curv_list: Optional[List[float]] = None
        if opts.field_curvature_influence > 0 and self.aggregator is not None:
            points = [t.end for t in tips]
            if hasattr(self.aggregator, "compute_field_curvature_many"):
                curv_list = self.aggregator.compute_field_curvature_many(points)
            else:
                curv_list = [self.aggregator.compute_field_curvature(p) for p in points]

        # --- RNG (per-tip deterministic) ---
        rand_vectors = None
        if opts.random_walk > 0:
            import numpy as _np
            rand_vectors = _np.empty((n, 3), dtype=float)
            for i, tip in enumerate(tips):
                seed64 = int(_combine_seed(master_seed, step, int(tip.id)))
                g = Generator(PCG64(seed64))
                rand_vectors[i, :] = g.normal(0.0, 1.0, 3)

        # --- per-tip combine (same math/order as _compute_with_rng) ---
        for i, tip in enumerate(tips):
            ori = out_orients[i]
            grad = grad_list[i]

            # autotropism + alignment + curvature
            if grad is not None:
                ori.add(grad.copy().scale(opts.autotropism))

                if opts.field_alignment_boost > 0:
                    gu = grad.copy().normalise()
                    dot = ori.dot(gu)
                    if dot > 0:
                        ori.add(gu.scale(dot * opts.field_alignment_boost))

                if opts.field_curvature_influence > 0 and curv_list is not None:
                    curvature = curv_list[i]
                    direction = grad.copy().normalise()
                    ori.add(direction.scale(curvature * opts.field_curvature_influence))

            # density avoidance
            if opts.die_if_too_dense and self.density_grid:
                ori.subtract(self.density_grid.get_gradient_at(tip.end))

            # gravitropism
            if opts.gravitropism > 0:
                z = tip.end.coords[2]
                if z < opts.gravi_angle_start:
                    strength = 0.0
                elif z > opts.gravi_angle_end:
                    strength = float(opts.gravitropism)
                else:
                    tlin = (z - opts.gravi_angle_start) / (opts.gravi_angle_end - opts.gravi_angle_start)
                    strength = float(tlin * opts.gravitropism)
                ori.add(MPoint(0, -1, 0).scale(strength))

            # nutrient sources
            for nutrient in self.nutrient_sources:
                delta = nutrient.copy().subtract(tip.end)
                dist = delta.magnitude()
                if dist < opts.nutrient_radius:
                    influence = 1.0 - (dist / opts.nutrient_radius)
                    if opts.nutrient_attraction > 0:
                        ori.add(delta.copy().normalise().scale(opts.nutrient_attraction * influence))
                    if opts.nutrient_repulsion > 0:
                        ori.subtract(delta.copy().normalise().scale(opts.nutrient_repulsion * influence))

            # anisotropy
            if opts.anisotropy_enabled:
                dir_vec = (
                    self.anisotropy_grid.get_direction_at(tip.end)
                    if self.anisotropy_grid else
                    MPoint(*opts.anisotropy_vector).normalise()
                )
                ori.add(dir_vec.scale(opts.anisotropy_strength))

            # random walk (from deterministic RNG batch)
            if opts.random_walk > 0 and rand_vectors is not None:
                rw = rand_vectors[i, :]
                ori.add(MPoint(*rw).normalise().scale(opts.random_walk))

            # memory blend
            if opts.direction_memory_blend > 0 and tip.orientation:
                blend = opts.direction_memory_blend
                ori = tip.orientation.copy().scale(blend).add(ori.scale(1.0 - blend)).normalise()

            out_orients[i] = ori.normalise()

        return out_orients
