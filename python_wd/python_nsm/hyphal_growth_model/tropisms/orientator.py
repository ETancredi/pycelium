# tropisms/orientator.py

# Imports
from __future__ import annotations
from typing import List, Optional
from core.point import MPoint  # 3D point/vector ops
from core.section import Section  # Section segments in mycelium
from core.options import Options  # Sim params
from compute.field_aggregator import FieldAggregator  # Aggregated multiple field sources
from vis.anisotropy_grid import AnisotropyGrid  # Grid-based anisotropy directions
import numpy as np  # Numerical utilities
from numpy.random import Generator, PCG64
import logging  # Logging tool
logger = logging.getLogger("pycelium")


def _stable_uint64(x: int) -> np.uint64:
    """
    SplitMix64-style scrambler to derive a stable 64-bit value from an int.
    Deterministic across platforms/threads.
    """
    z = (np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
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
    """Combines multiple tropism influences to compute new growth orientation."""
    def __init__(self, options: Options):
        self.options = options  # Store global sim opts
        self.aggregator: FieldAggregator = None  # Placeholder for FieldAggregator instance
        self.density_grid = None  # Placeholder for density grid (avoidance)
        self.anisotropy_grid: AnisotropyGrid = None  # Placeholder for anisotropy grid
        self.nutrient_sources: List[MPoint] = []  # List of nutrient source points (MPoint instances)

    def set_field_source(self, aggregator: FieldAggregator):
        self.aggregator = aggregator  # Assign the FieldAggregator for chemical/substrate fields

    def set_density_grid(self, grid):
        self.density_grid = grid  # Assign a density grid object that provides avoidance gradients

    def set_anisotropy_grid(self, grid: AnisotropyGrid):
        self.anisotropy_grid = grid  # Assign an anisotropy grid for directional bias

    def set_nutrient_sources(self, points: List[MPoint]):
        self.nutrient_sources = points  # Set list of nutrient attractor/repellent points

    # Deterministic entrypoint for parallel-safe orientation computation
    def compute_deterministic(self, section: Section, *, step: int, master_seed: Optional[int]) -> MPoint:
        """
        Deterministic, thread-safe orientation compute:
        - Creates a local RNG seeded by (master_seed, step, section.id).
        - No global RNG use or shared-state mutation.
        - Returns exactly the same result as `compute()` would have produced
          if `compute()` used the same seed and math.
        """
        seed64 = int(_combine_seed(master_seed, step, int(section.id)))
        rng = Generator(PCG64(seed64))
        return self._compute_with_rng(section, rng)

    # Internal helper that mirrors `compute()` but uses a provided RNG.
    # All stochastic draws MUST come from `rng` (no np.random/random).
    def _compute_with_rng(self, section: Section, rng: Generator) -> MPoint:
        opts = self.options
        orientation = section.orientation.copy()  # Start w/ current orientation as a mutable copy

        # Autotropism & Field Alignment
        grad = None
        if self.aggregator:
            _, grad = self.aggregator.compute_field(section.end)  # field + gradient at section end
            if grad is not None:
                orientation.add(grad.scale(opts.autotropism))  # along gradient

                # Boost alignment with field gradient
                if opts.field_alignment_boost > 0:
                    grad_unit = grad.copy().normalise()
                    dot = orientation.dot(grad_unit)
                    if dot > 0:
                        boost = grad_unit.scale(dot * opts.field_alignment_boost)
                        orientation.add(boost)
                        logger.debug(f"Gradient alignment boost: dot={dot:.2f}, boost={boost}")

                # Curvature influence from field
                if opts.field_curvature_influence > 0:
                    curvature = self.aggregator.compute_field_curvature(section.end)  # Laplacian approx
                    direction = grad.copy().normalise()
                    orientation.add(direction.scale(curvature * opts.field_curvature_influence))
                    logger.debug(
                        f"Curvature contribution: value={curvature:.3f}, "
                        f"scaled={curvature * opts.field_curvature_influence:.3f}"
                    )

        # Density-based avoidance
        if opts.die_if_too_dense and self.density_grid:
            density_grad = self.density_grid.get_gradient_at(section.end)  # Points toward higher density
            orientation.subtract(density_grad)  # Steer away from high-density regions

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
            gravity_vec = MPoint(0, -1, 0).scale(strength)  # Downward in Y
            orientation.add(gravity_vec)

        # Nutrient fields
        for nutrient in self.nutrient_sources:
            delta = nutrient.copy().subtract(section.end)  # Vector toward nutrient source
            dist = delta.magnitude()
            if dist < opts.nutrient_radius:
                influence = 1.0 - (dist / opts.nutrient_radius)
                if opts.nutrient_attraction > 0:
                    orientation.add(delta.copy().normalise().scale(opts.nutrient_attraction * influence))
                if opts.nutrient_repulsion > 0:
                    orientation.subtract(delta.copy().normalise().scale(opts.nutrient_repulsion * influence))

        # Global or Grid-Based Anisotropy
        if opts.anisotropy_enabled:
            if self.anisotropy_grid:
                dir_vec = self.anisotropy_grid.get_direction_at(section.end)
                logger.debug(f"Grid-based anisotropy: {dir_vec}")
            else:
                dir_vec = MPoint(*opts.anisotropy_vector).normalise()
                logger.debug(f"Global anisotropy: {opts.anisotropy_vector}")
            orientation.add(dir_vec.scale(opts.anisotropy_strength))

        # Random walk (use provided RNG)
        if opts.random_walk > 0:
            rand = rng.normal(0.0, 1.0, 3)
            orientation.add(MPoint(*rand).normalise().scale(opts.random_walk))

        # Directional memory blending
        if opts.direction_memory_blend > 0 and section.orientation:
            blend = opts.direction_memory_blend
            orientation = (
                section.orientation.copy().scale(blend)
                .add(orientation.scale(1.0 - blend))
                .normalise()
            )
            logger.debug(f"Orientation memory: blend={blend:.2f}")

        return orientation.normalise()
    
    # Legacy API (unchanged) — single-tip compute using global RNG.
    # Kept for single-threaded paths & backwards compatibility.
    def compute(self, section: Section) -> MPoint:
        """
        Original API: uses global RNG (np.random). Not thread-safe deterministic.
        """
        orientation = section.orientation.copy()  # Start w/ current orientation as a mutable copy

        # Autotropism & Field Alignment
        grad = None

        if self.aggregator:
            _, grad = self.aggregator.compute_field(section.end)  # Compute scalar field and gradient at section end
            if grad is not None:
                orientation.add(grad.scale(self.options.autotropism))  # Push orientation slightly along gradient direction

                # Boost alignment with field gradient
                if self.options.field_alignment_boost > 0:
                    grad_unit = grad.copy().normalise()  # Unit gradient vector
                    dot = orientation.dot(grad_unit)  # cos(angle) between orientation & gradient
                    if dot > 0:
                        boost = grad_unit.scale(dot * self.options.field_alignment_boost)
                        orientation.add(boost)
                        logger.debug(f"Gradient alignment boost: dot={dot:.2f}, boost={boost}")

                # Curvature influence from field
                if self.options.field_curvature_influence > 0:
                    curvature = self.aggregator.compute_field_curvature(section.end)  # Approximate Laplacian of scalar field
                    direction = grad.copy().normalise()  # Unit direction of gradient
                    orientation.add(direction.scale(curvature * self.options.field_curvature_influence))
                    logger.debug(f"Curvature contribution: value={curvature:.3f}, scaled={curvature * self.options.field_curvature_influence:.3f}")

        # Density-based avoidance
        if self.options.die_if_too_dense and self.density_grid:
            density_grad = self.density_grid.get_gradient_at(section.end)  # Points toward higher density
            orientation.subtract(density_grad)  # Steer away from high-density regions

        # Gravitropism
        if self.options.gravitropism > 0:
            z = section.end.coords[2]  # Current height (Z)
            if z < self.options.gravi_angle_start:
                strength = 0
            elif z > self.options.gravi_angle_end:
                strength = self.options.gravitropism
            else:
                # Interpolate between start and end heights
                t = (z - self.options.gravi_angle_start) / (self.options.gravi_angle_end - self.options.gravi_angle_start)
                strength = t * self.options.gravitropism
            gravity_vec = MPoint(0, -1, 0).scale(strength)  # Downward in Y
            orientation.add(gravity_vec)

        # Nutrient fields
        for nutrient in self.nutrient_sources:
            delta = nutrient.copy().subtract(section.end)  # Vector toward nutrient source
            dist = delta.magnitude()
            if dist < self.options.nutrient_radius:
                influence = 1.0 - (dist / self.options.nutrient_radius)
                if self.options.nutrient_attraction > 0:
                    orientation.add(delta.copy().normalise().scale(self.options.nutrient_attraction * influence))
                if self.options.nutrient_repulsion > 0:
                    orientation.subtract(delta.copy().normalise().scale(self.options.nutrient_repulsion * influence))

        # Global or Grid-Based Anisotropy
        if self.options.anisotropy_enabled:
            if self.anisotropy_grid:
                dir_vec = self.anisotropy_grid.get_direction_at(section.end)
                logger.debug(f"Grid-based anisotropy: {dir_vec}")
            else:
                dir_vec = MPoint(*self.options.anisotropy_vector).normalise()
                logger.debug(f"Global anisotropy: {self.options.anisotropy_vector}")
            orientation.add(dir_vec.scale(self.options.anisotropy_strength))

        # Random walk
        if self.options.random_walk > 0:
            rand = np.random.normal(0, 1, 3)
            orientation.add(MPoint(*rand).normalise().scale(self.options.random_walk))

        # Directional memory blending
        if self.options.direction_memory_blend > 0 and section.orientation:
            blend = self.options.direction_memory_blend
            orientation = (
                section.orientation.copy().scale(blend)
                .add(orientation.scale(1.0 - blend))
                .normalise()
            )
            logger.debug(f"Orientation memory: blend={blend:.2f}")

        return orientation.normalise()  # Ensure final orientation is a unit vector

    # NEW (kept as you provided): batched compute using aggregator.compute_field_many()
    def compute_many(self, tips: List[Section]) -> List[MPoint]:
        """
        Compute new orientations for a list of tips, preserving order and results.
        Falls back cleanly if aggregator is missing (then identical to per-tip).
        """
        n = len(tips)
        if n == 0:
            return []

        # Start from current orientations
        out_orients = [t.orientation.copy() for t in tips]

        # ---- fields & gradients in one batched call (preserves order) ----
        grad_list = [None] * n
        if self.aggregator is not None:
            points = [t.end for t in tips]
            _, grads = self.aggregator.compute_field_many(points)
            grad_list = grads  # list of MPoint (unit)

        # Now apply the exact same influences as in `compute`, per tip.
        opts = self.options
        for i, tip in enumerate(tips):
            ori = out_orients[i]
            grad = grad_list[i]

            # Autotropism & field alignment
            if grad is not None:
                ori.add(grad.copy().scale(opts.autotropism))

                if opts.field_alignment_boost > 0:
                    grad_unit = grad.copy().normalise()
                    dot = ori.dot(grad_unit)
                    if dot > 0:
                        boost = grad_unit.scale(dot * opts.field_alignment_boost)
                        ori.add(boost)

                if opts.field_curvature_influence > 0 and self.aggregator is not None:
                    curvature = self.aggregator.compute_field_curvature(tip.end)
                    direction = grad.copy().normalise()
                    ori.add(direction.scale(curvature * opts.field_curvature_influence))

            # Density-based avoidance
            if opts.die_if_too_dense and self.density_grid:
                density_grad = self.density_grid.get_gradient_at(tip.end)
                ori.subtract(density_grad)

            # Gravitropism
            if opts.gravitropism > 0:
                z = tip.end.coords[2]
                if z < opts.gravi_angle_start:
                    strength = 0
                elif z > opts.gravi_angle_end:
                    strength = opts.gravitropism
                else:
                    t = (z - opts.gravi_angle_start) / (opts.gravi_angle_end - opts.gravi_angle_start)
                    strength = t * opts.gravitropism
                ori.add(MPoint(0, -1, 0).scale(strength))

            # Nutrient fields
            for nutrient in self.nutrient_sources:
                delta = nutrient.copy().subtract(tip.end)
                dist = delta.magnitude()
                if dist < opts.nutrient_radius:
                    influence = 1.0 - (dist / opts.nutrient_radius)
                    if opts.nutrient_attraction > 0:
                        ori.add(delta.copy().normalise().scale(opts.nutrient_attraction * influence))
                    if opts.nutrient_repulsion > 0:
                        ori.subtract(delta.copy().normalise().scale(opts.nutrient_repulsion * influence))

            # Anisotropy
            if opts.anisotropy_enabled:
                if self.anisotropy_grid:
                    dir_vec = self.anisotropy_grid.get_direction_at(tip.end)
                else:
                    dir_vec = MPoint(*opts.anisotropy_vector).normalise()
                ori.add(dir_vec.scale(opts.anisotropy_strength))

            # Random walk (legacy global RNG)
            if opts.random_walk > 0:
                rand = np.random.normal(0, 1, 3)
                ori.add(MPoint(*rand).normalise().scale(opts.random_walk))

            # Directional memory blending
            if opts.direction_memory_blend > 0 and tip.orientation:
                blend = opts.direction_memory_blend
                ori = tip.orientation.copy().scale(blend).add(ori.scale(1.0 - blend)).normalise()

            out_orients[i] = ori.normalise()

        return out_orients
