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

    def compute_local_hyphal_density(self, radius: float) -> float:
        """
        Continuous crowding metric: sum the decaying-field contributions 
        of all other hyphal segments within `radius`.
        """
        density = 0.0
        for src in self.field_aggregator.sources:
            if not isinstance(src, SectFieldFinder):
                continue
            if src.section is self:
                continue
            if self.end.distance_to(src.section.end) > radius:
                continue
            density += src.find_field(self.end)
        return density

    def grow(self, rate: float, dt: float):
        if not self.is_tip or self.is_dead:
            return

        # --- Tropism blending into self.orientation ---
        net = MPoint(0, 0, 0)

        # 1. Autotropism along own axis
        if self.options.autotropism != 0:
            net.add(
                self.orientation.copy()
                                .scale(self.options.autotropism * self.options.autotropism_impact)
            )

        # 2. FieldHypothesis toward/away from field gradient
        if self.options.field_hypothesis and self.field_aggregator:
            strength, grad = self.field_aggregator.compute_field(
                self.end, exclude_ids=[id(self)]
            )
            net.add(grad.copy().scale(strength))

        # 3. Gravitropism: constant pull along –Z
        if self.options.gravitropism != 0:
            net.add(MPoint(0, 0, -1).scale(self.options.gravitropism))

        # 4. Random walk jitter
        if self.options.random_walk > 0:
            rv = MPoint(*np.random.randn(3)).normalise().scale(self.options.random_walk)
            net.add(rv)

        # Apply new orientation if any tropism contributed
        if np.linalg.norm(net.coords) > 0:
            self.orientation = net.normalise()
        # --- End tropism blending ---
        
        # Length‐scaled growth
        if self.options.length_scaled_growth:
            scale_factor = 1 + self.length * self.options.length_growth_coef
            rate *= scale_factor

        # Age‐based slowdown (d_age)
        age = self.age if self.age > 0 else 1.0
        rate /= (age ** self.options.d_age)

        # Default growth‐vector scaling
        rate *= self.options.default_growth_vector

        # Actual growth step
        growth_distance = rate * dt
        delta = self.orientation.copy().scale(growth_distance)
        prev_end = self.end.copy()
        self.end.add(delta)
        self.length += growth_distance
        self.age += dt
        self.subsegments.append((prev_end, self.end.copy()))

        # EMA‐style directional memory update
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

        # 1. Master on/off switch
        if not self.options.branching_master:
            return None

        # 2. Cap on branches from this node
        if self.branches_made >= self.options.max_branches:
            return None

        # 3. Legacy‐mode cutoff
        if self.options.old_nbranch and self.age > self.options.branch_time_window:
            return None

        # 4. Only allow secondary branches if enabled
        if self.children and not self.options.secondary_branching:
            return None

        # 5. Hyphal‐crowding gate
        if self.options.density_dependend:
            local_density = self.compute_local_hyphal_density(self.options.neighbour_radius)
            if local_density >= self.options.branching_density:
                return None

        # 6. Environmental‐field gate (max over tip/subsegments/branch point)
        if self.field_aggregator:
            points = (
                [end for _, end in self.subsegments]
                if self.options.complete_evaluation
                else [self.end]
            )
            if self.options.log_branch_points and len(self.subsegments) > 1:
                points.append(self.subsegments[1][0])

            strengths = [self.field_aggregator.compute_field(pt, exclude_ids=[id(self)])[0]
                         for pt in points]
            if max(strengths) >= self.options.field_threshold:
                return None

        # 7. Random gate
        if np.random.rand() >= self.options.branch_probability:
            return None

        # 8. Determine branch orientation via 3D helper
        rotated_orientation = self.get_new_growing_vector(
            self.options.default_growth_vector
        )

        # 🌀 Curvature bias (unchanged)
        if self.options.curvature_branch_bias > 0 and len(self.subsegments) >= 3:
            p1 = self.subsegments[-3][0]
            p2 = self.subsegments[-2][0]
            p3 = self.subsegments[-1][1]
            v1 = p2.copy().subtract(p1).normalise()
            v2 = p3.copy().subtract(p2).normalise()
            curve = v2.copy().subtract(v1).normalise()
            rotated_orientation = (
                rotated_orientation.copy().scale(1 - self.options.curvature_branch_bias)
                                    .add(curve.copy().scale(self.options.curvature_branch_bias))
                                    .normalise()
            )

        # 🧠 Directional‐memory bias (unchanged)
        if self.options.direction_memory_blend > 0:
            rotated_orientation = (
                rotated_orientation.copy().scale(1 - self.options.direction_memory_blend)
                                    .add(self.direction_memory.copy().scale(self.options.direction_memory_blend))
                                    .normalise()
            )

        # 9. Leading‐branch selection
        keep_self_leading = (np.random.rand() < self.options.leading_branch_prob)
        if keep_self_leading:
            child_orientation = rotated_orientation
        else:
            child_orientation = self.orientation.copy()
            self.orientation = rotated_orientation

        # Instantiate and return the new branch
        child = Section(self.end.copy(), child_orientation, parent=self)
        child.is_tip = True
        child.options = self.options
        child.direction_memory = self.direction_memory.copy()
        child.set_field_aggregator(self.field_aggregator)

        self.children.append(child)
        self.branches_made += 1
        return child

    def get_new_growing_vector(self, default_strength: float):
        parent = self.orientation.copy().normalise()
        max_angle_rad = math.radians(self.options.branch_angle_spread)

        # 1) Optimal orientation via field
        if self.options.optimal_branch_orientation and self.field_aggregator:
            strength, field_vec = self.field_aggregator.compute_field(
                self.end, exclude_ids=[id(self)]
            )
            if field_vec.length() > 0:
                new_dir = field_vec.copy().normalise().scale(default_strength)
            else:
                new_dir = None
        else:
            new_dir = None

        # 2) Uniform 3D cone sampling (if no optimal vector)
        if new_dir is None:
            cos_max = math.cos(max_angle_rad)
            u = np.random.uniform(cos_max, 1.0)
            phi = math.acos(u)
            theta = np.random.uniform(0, 2 * math.pi)

            # Build orthonormal basis (parent, perp1, perp2)
            if abs(parent.coords[0]) < 0.9:
                temp = MPoint(1, 0, 0)
            else:
                temp = MPoint(0, 1, 0)
            perp1 = parent.cross(temp).normalise()
            perp2 = parent.cross(perp1).normalise()
            part1 = parent.copy().scale(math.cos(phi))
            part2 = perp1.copy().scale(math.sin(phi) * math.cos(theta))
            part3 = perp2.copy().scale(math.sin(phi) * math.sin(theta))
            new_dir = part1.add(part2).add(part3).normalise().scale(default_strength)

        # --- Plagiotropism tolerance ---
        # Clamp to pure downward if too far from vertical
        tol_rad = math.radians(self.options.plagiotropism_tolerance_angle)
        # true “down” is –Z
        vertical = MPoint(0, 0, -1).normalise()
        cosang = new_dir.dot(vertical)
        cosang = max(-1.0, min(1.0, cosang))
        if math.acos(cosang) > tol_rad:
            new_dir = vertical.scale(default_strength)

        return new_dir
    def get_subsegments(self):
        return [(s.copy(), e.copy()) for s, e in self.subsegments]

    def __str__(self):
        status = "DEAD" if self.is_dead else ("TIP" if self.is_tip else "BRANCHED")
        return f"[{status}] {self.start} -> {self.end} | len={self.length:.2f}"
