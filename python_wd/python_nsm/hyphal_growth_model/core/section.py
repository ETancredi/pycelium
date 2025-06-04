# core/section.py

import math
import numpy as np
from core.point import MPoint
from tropisms.sect_field_finder import SectFieldFinder

class Section:
    """Represents a single hyphal segment (tip or branch) in the fungal network"""

    def __init__(self, start: MPoint, orientation: MPoint, parent=None):
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

        self.options = None
        self.field_aggregator = None

        self.direction_memory = self.orientation.copy()  

    def set_field_aggregator(self, aggregator):
        self.field_aggregator = aggregator

    def grow(self, rate: float, dt: float):
        if not self.is_tip or self.is_dead:
            return

        if self.options and self.options.length_scaled_growth:
            scale_factor = 1 + self.length * self.options.length_growth_coef
            rate *= scale_factor

        growth_distance = rate * dt
        delta = self.orientation.copy().scale(growth_distance)
        prev_end = self.end.copy()
        self.end.add(delta)
        self.length += growth_distance
        self.age += dt
        self.subsegments.append((prev_end, self.end.copy()))

        # Volume Constraint Check (tip stops at boundary)
        opts = self.options
        if opts.volume_constraint:
            x, y, z = self.end.coords
            
            # If any coordinate is outside, clamp and stop future growth:
            out_of_bounds = False
            
            # X-axis:
            if x < opts.x_min:
                x = opts.x_min
                out_of_bounds = True
            elif x > opts.x_max:
                x = opts.x_max
                out_of_bounds = True

            # Y-axis:
            if y < opts.y_min:
                y = opts.y_min
                out_of_bounds = True
            elif y > opts.y_max:
                y = opts.y_max
                out_of_bounds = True
            
            # Z-axis:
            if z < opts.z_min:
                z = opts.z_min
                out_of_bounds = True
            elif z > opts.z_max:
                z = opts.z_max
                out_of_bounds = True

            if out_of_bounds:
                # Clamp the offending tip to the fit it hits:
                self.end = MPoint(x, y, z)
                # Recompute length so that the segment does not extend past the box:
                self.length = self.start.distance_to(self.end)
                # Inactivate the tip, so it will not continue to grow
                self.is_tip = False
                return

        # Update directional memory (EMA-style)
        if self.options and hasattr(self.options, "direction_memory_blend"):
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

    def maybe_branch(self, branch_chance: float, tip_count: int = 0):
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

        if np.random.rand() < branch_chance:
            angle = np.random.uniform(-self.options.branch_angle_spread, self.options.branch_angle_spread)
            axis = MPoint(0, 0, 1)
            rotated_orientation = self.orientation.copy().rotated_around(axis, angle)

            # 🌀 Curvature bias
            if self.options.curvature_branch_bias > 0 and len(self.subsegments) >= 3:
                p1 = self.subsegments[-3][0]
                p2 = self.subsegments[-2][0]
                p3 = self.subsegments[-1][1]
                v1 = p2.copy().subtract(p1).normalise()
                v2 = p3.copy().subtract(p2).normalise()
                curve = v2.copy().subtract(v1).normalise()

                rotated_orientation = (
                    rotated_orientation.copy().scale(1.0 - self.options.curvature_branch_bias)
                    .add(curve.copy().scale(self.options.curvature_branch_bias))
                    .normalise()
                )
                print(f"🌀 Curvature blended into branch direction: strength={self.options.curvature_branch_bias:.2f}")

            # 🧠 Memory-based bias
            if self.options.direction_memory_blend > 0:
                rotated_orientation = (
                    rotated_orientation.copy().scale(1.0 - self.options.direction_memory_blend)
                    .add(self.direction_memory.copy().scale(self.options.direction_memory_blend))
                    .normalise()
                )
                print(f"🧠 Directional memory blended into branch orientation: alpha={self.options.direction_memory_blend:.2f}")

            keep_self_leading = np.random.rand() < self.options.leading_branch_prob
            if keep_self_leading:
                child_orientation = rotated_orientation
            else:
                child_orientation = self.orientation.copy()
                self.orientation = rotated_orientation

            child = Section(self.end.copy(), child_orientation, parent=self)
            child.is_tip = True
            child.options = self.options
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
