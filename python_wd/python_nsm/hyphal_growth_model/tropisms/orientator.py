# tropisms/orientator.py

from core.point import MPoint
from core.section import Section
from core.options import Options
from compute.field_aggregator import FieldAggregator
from vis.anisotropy_grid import AnisotropyGrid  
import numpy as np
import math

class Orientator:
    def __init__(self, options: Options):
        self.options = options
        self.aggregator: FieldAggregator = None
        self.density_grid = None
        self.anisotropy_grid: AnisotropyGrid = None  
        self.nutrient_sources: list[MPoint] = []

    def set_field_source(self, aggregator: FieldAggregator):
        self.aggregator = aggregator

    def set_density_grid(self, grid):
        self.density_grid = grid

    def set_anisotropy_grid(self, grid: AnisotropyGrid):
        self.anisotropy_grid = grid

    def set_nutrient_sources(self, points: list[MPoint]):
        self.nutrient_sources = points

    def compute(self, section: Section) -> MPoint:
        orientation = section.orientation.copy()

        # Negative Autotropism
        grad = None
        if self.aggregator:
            _, grad = self.aggregator.compute_field(section.end)
            if grad is not None:
                orientation.add(grad.scale(self.options.negative_autotropism))

                # Boost alignment with field gradient
                if self.options.field_alignment_boost > 0:
                    grad_unit = grad.copy().normalise()
                    dot = orientation.dot(grad_unit)
                    if dot > 0:
                        boost = grad_unit.scale(dot * self.options.field_alignment_boost)
                        orientation.add(boost)
                        print(f"Gradient alignment boost applied: dot={dot:.2f}, boost={boost}")

                # Curvature influence from field
                if self.options.field_curvature_influence > 0:
                    curvature = self.aggregator.compute_field_curvature(section.end)
                    direction = grad.copy().normalise()
                    orientation.add(direction.scale(curvature * self.options.field_curvature_influence))
                    print(f"🌀 Curvature contribution: value={curvature:.3f}, scaled={curvature * self.options.field_curvature_influence:.3f}")

        # Density-based avoidance
        if self.options.die_if_too_dense and self.density_grid:
            density_grad = self.density_grid.get_gradient_at(section.end)
            orientation.subtract(density_grad)

        # Gravitropism
        if self.options.gravitropism > 0:
            z = section.end.coords[2]
            if z < self.options.gravi_angle_start:
                strength = 0
            elif z > self.options.gravi_angle_end:
                strength = self.options.gravitropism
            else:
                t = (z - self.options.gravi_angle_start) / (self.options.gravi_angle_end - self.options.gravi_angle_start)
                strength = t * self.options.gravitropism
            gravity_vec = MPoint(0, -1, 0).scale(strength)
            orientation.add(gravity_vec)

        # Nutrient fields
        for nutrient in self.nutrient_sources:
            delta = nutrient.copy().subtract(section.end)
            dist = delta.magnitude()

            if dist < self.options.nutrient_radius:
                influence = 1.0 - (dist / self.options.nutrient_radius)

                if self.options.nutrient_attraction > 0:
                    orientation.add(
                        delta.copy().normalise().scale(self.options.nutrient_attraction * influence)
                    )
                if self.options.nutrient_repulsion > 0:
                    orientation.subtract(
                        delta.copy().normalise().scale(self.options.nutrient_repulsion * influence)
                    )

        # Global or Grid-Based Anisotropy
        if self.options.anisotropy_enabled:
            if self.anisotropy_grid:
                dir_vec = self.anisotropy_grid.get_direction_at(section.end)
                print(f"🧭 Grid-based anisotropy: {dir_vec}")
            else:
                dir_vec = MPoint(*self.options.anisotropy_vector).normalise()
                print(f"🧭 Global anisotropy: {self.options.anisotropy_vector}")

            orientation.add(dir_vec.scale(self.options.anisotropy_strength))

        # Random walk
        if self.options.random_walk > 0:
            rand = np.random.normal(0, 1, 3)
            orientation.add(MPoint(*rand).normalise().scale(self.options.random_walk))

        # Directional memory blending (EMA-style)
        if self.options.direction_memory_blend > 0 and section.orientation:
            blend = self.options.direction_memory_blend
            orientation = (
                section.orientation.copy().scale(blend)
                .add(orientation.scale(1.0 - blend))
                .normalise()
            )
            print(f"🧠 Orientation memory applied: blend={blend:.2f}")

        return orientation.normalise()
