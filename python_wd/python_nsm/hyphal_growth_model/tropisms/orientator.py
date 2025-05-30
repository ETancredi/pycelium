# tropisms/orientator.py

import math
import numpy as np
from core.point import MPoint
from core.options import Options, ToggleableFloat, ToggleableInt
from core.section import Section
from compute.field_aggregator import FieldAggregator
from vis.anisotropy_grid import AnisotropyGrid  

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

        # 1) Autotropism (self-avoidance) via field gradient
        if self.aggregator:
            _, grad = self.aggregator.compute_field(section.end)
            if grad is not None:
                # unwrap autotropism & impact
                at = self.options.autotropism
                ai = self.options.autotropism_impact
                if isinstance(at, ToggleableFloat) and isinstance(ai, ToggleableFloat):
                    if at.enabled and ai.enabled and at.value != 0 and ai.value != 0:
                        orientation.add(grad.copy().scale(at.value * ai.value))
                else:
                    # legacy floats
                    orientation.add(grad.copy().scale(self.options.autotropism * self.options.autotropism_impact))

                # 1a) Boost alignment with field gradient
                fab = self.options.field_alignment_boost
                if isinstance(fab, ToggleableFloat):
                    boost_strength = fab.value if fab.enabled else 0.0
                else:
                    boost_strength = fab
                if boost_strength > 0:
                    grad_unit = grad.copy().normalise()
                    dot = orientation.dot(grad_unit)
                    if dot > 0:
                        orientation.add(grad_unit.scale(dot * boost_strength))

                # 1b) Curvature influence from field
                fci = self.options.field_curvature_influence
                if isinstance(fci, ToggleableFloat):
                    ci_strength = fci.value if fci.enabled else 0.0
                else:
                    ci_strength = fci
                if ci_strength > 0:
                    curvature = self.aggregator.compute_field_curvature(section.end)
                    direction = grad.copy().normalise()
                    orientation.add(direction.scale(curvature * ci_strength))

        # 2) Density‐grid avoidance
        if self.options.die_if_too_dense and self.density_grid:
            density_grad = self.density_grid.get_gradient_at(section.end)
            orientation.subtract(density_grad)

        # 3) Gravitropism (ramped between start/end)
        gr = self.options.gravitropism
        # unwrap tog‐float vs legacy
        if isinstance(gr, ToggleableFloat):
            strength_val = gr.value if gr.enabled else 0.0
        else:
            strength_val = gr
        if strength_val > 0:
            z = section.end.coords[2]
            if z < self.options.gravi_angle_start:
                s = 0.0
            elif z > self.options.gravi_angle_end:
                s = strength_val
            else:
                t = ((z - self.options.gravi_angle_start) /
                     (self.options.gravi_angle_end - self.options.gravi_angle_start))
                s = t * strength_val
            orientation.add(MPoint(0, 0, -1).scale(s))

        # 4) Nutrient fields (unchanged)
        for nutrient in self.nutrient_sources:
            delta = nutrient.copy().subtract(section.end)
            dist = np.linalg.norm(delta.as_array())
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

        # 5) Anisotropy
        if self.options.anisotropy_enabled:
            if self.anisotropy_grid:
                dir_vec = self.anisotropy_grid.get_direction_at(section.end)
            else:
                dir_vec = MPoint(*self.options.anisotropy_vector).normalise()
            orientation.add(dir_vec.scale(self.options.anisotropy_strength))

        # 6) Random walk jitter
        if self.options.random_walk > 0:
            rv = MPoint(*np.random.randn(3)).normalise().scale(self.options.random_walk)
            orientation.add(rv)

        # 7) Directional‐memory blending (EMA‐style)
        dmb = self.options.direction_memory_blend
        if isinstance(dmb, ToggleableFloat):
            blend = dmb.value if dmb.enabled else 0.0
        else:
            blend = dmb
        if blend > 0:
            orientation = (
                section.orientation.copy().scale(blend)
                .add(orientation.scale(1.0 - blend))
                .normalise()
            )

        return orientation.normalise()
