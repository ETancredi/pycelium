# core/section.py

import math
import numpy as np
import random
import itertools
from core.point import MPoint
from core.options import Options
from core.phenotypes import Phenotype
from typing import Optional, Tuple

# Global counter for unique Section IDs
_SECTION_ID_GEN = itertools.count()

class Section:
    """Represents a single hyphal segment (tip or branch) in the fungal network"""

    def __init__(
        self, 
        start: MPoint, 
        orientation: MPoint, 
        opts: Options, 
        parent: Optional["Section"] = None, 
        color: Optional[Tuple[float, float, float]] = None,
        phenotype: Optional[Phenotype] = None
    ):
        self.id = next(_SECTION_ID_GEN)
        self.start = start.copy()
        self.orientation = orientation.copy().normalise()
        self.length = 0.0
        self.age = 0.0

        self.is_tip = True
        self.is_dead = False
        self.branches_made = 0

        self.parent = parent
        self.children = []

        self.end = self.start.copy()
        self.subsegments = [(self.start.copy(), self.start.copy())]

        self.options = opts
        self.field_aggregator = None

        self.direction_memory = self.orientation.copy() 
        # Initialise color
        if parent is None:
            self.color = color if color is not None else opts.initial_color
        else:
            self.color = color

        # Attach phenotype
        self.phenotype = phenotype

    def set_field_aggregator(self, aggregator):
        self.field_aggregator = aggregator

    def grow(self, rate: float, dt: float):
        if not self.is_tip or self.is_dead:
            return

        if self.options.length_scaled_growth:
            scale_factor = 1 + self.length * self.options.length_growth_coef
            rate *= scale_factor

        growth_distance = rate * dt
        delta = self.orientation.copy().scale(growth_distance)
        prev_end = self.end.copy()
        self.end.add(delta)
        self.length += growth_distance
        self.age += dt
        self.subsegments.append((prev_end, self.end.copy()))

        # Volume Constraint Check
        if self.options.volume_constraint:
            x, y, z = self.end.coords
            out_of_bounds = False
            if x < self.options.x_min:
                x = self.options.x_min; out_of_bounds = True
            elif x > self.options.x_max:
                x = self.options.x_max; out_of_bounds = True
            if y < self.options.y_min:
                y = self.options.y_min; out_of_bounds = True
            elif y > self.options.y_max:
                y = self.options.y_max; out_of_bounds = True
            if z < self.options.z_min:
                z = self.options.z_min; out_of_bounds = True
            elif z > self.options.z_max:
                z = self.options.z_max; out_of_bounds = True

            if out_of_bounds:
                self.end = MPoint(x, y, z)
                self.length = self.start.distance_to(self.end)
                self.is_tip = False
                return

        # Update directional memory blending
        alpha = self.options.direction_memory_blend
        self.direction_memory = (
            self.direction_memory.scale(1 - alpha)
            .add(self.orientation.copy().scale(alpha))
            .normalise()
        )

    def update(self):
        self.length = self.start.distance_to(self.end)
        if self.length < 1e-5:
            self.is_dead = True

    def maybe_branch(self, branch_chance: float, tip_count: int = 0) -> Optional["Section"]:
        if not self.is_tip or self.is_dead:
            return None
        if self.branches_made >= self.options.max_branches:
            return None
        if self.age < self.options.min_tip_age or self.length < self.options.min_tip_length:
            return None
        if self.age > self.options.branch_time_window:
            return None

        if self.field_aggregator:
            field_strength, _ = self.field_aggregator.compute_field(self.end, exclude_ids=[id(self)])
            if field_strength >= self.options.field_threshold:
                return None

        if random.random() < branch_chance:
            # Determine new orientation
            angle = random.uniform(-self.options.branch_angle_spread, self.options.branch_angle_spread)
            axis = MPoint(0, 0, 1)
            rotated = self.orientation.copy().rotated_around(axis, angle)

            # Curvature bias
            if self.options.curvature_branch_bias > 0 and len(self.subsegments) >= 3:
                p1 = self.subsegments[-3][0]
                p2 = self.subsegments[-2][0]
                p3 = self.subsegments[-1][1]
                v1 = p2.copy().subtract(p1).normalise()
                v2 = p3.copy().subtract(p2).normalise()
                curve = v2.copy().subtract(v1).normalise()
                rotated = (
                    rotated.copy().scale(1 - self.options.curvature_branch_bias)
                    .add(curve.copy().scale(self.options.curvature_branch_bias))
                    .normalise()
                )

            # Directional memory blend
            if self.options.direction_memory_blend > 0:
                rotated = (
                    rotated.copy().scale(1 - self.options.direction_memory_blend)
                    .add(self.direction_memory.copy().scale(self.options.direction_memory_blend))
                    .normalise()
                )

            # Choose which branch keeps original orientation
            if random.random() < self.options.leading_branch_prob:
                child_orient = rotated
            else:
                child_orient = self.orientation.copy()
                self.orientation = rotated

            # Mutate phenotype with global settings
            child_pheno = self.phenotype.copy_with_mutation(
                mutation_scale=self.options.mutation_scale
            )
            child_color = child_pheno.color

            # Create child section with mutated phenotype
            child = Section(
                start=self.end.copy(),
                orientation=child_orient,
                opts=self.options,
                parent=self,
                color=child_color,
                phenotype=child_pheno
            )
            child.is_tip = True
            child.direction_memory = self.direction_memory.copy()
            child.set_field_aggregator(self.field_aggregator)

            self.children.append(child)
            self.branches_made += 1
            return child

        return None

    def get_subsegments(self):
        return [(s.copy(), e.copy()) for s, e in self.subsegments]

    def __str__(self):
        status = "DEAD" if self.is_dead else ("TIP" if self.is_tip else "BRANCHED")
        return f"[{status}] {self.start} -> {self.end} | len={self.length:.2f}"
