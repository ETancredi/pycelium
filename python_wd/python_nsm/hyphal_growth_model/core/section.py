# core/section.py

from core.point import MPoint
import numpy as np

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
    
        # length-scaled growth
        if self.options and self.options.length_scaled_growth:
            scale_factor = 1 + self.length * self.options.length_growth_coef
            rate *= scale_factor
    
        # age-based slowdown (d_age)
        if self.options and hasattr(self.options, "d_age"):
            age = self.age if self.age > 0 else 1.0
            rate /= (age ** self.options.d_age)
    
        # default growth vector scaling
        if self.options and hasattr(self.options, "default_growth_vector"):
            rate *= self.options.default_growth_vector
    
        # now compute actual growth
        growth_distance = rate * dt
        delta = self.orientation.copy().scale(growth_distance)
    
        prev_end = self.end.copy()
        self.end.add(delta)
        self.length += growth_distance
        self.age += dt
    
        self.subsegments.append((prev_end, self.end.copy()))
    
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
    
        # 1. Master switch for ALL branching
        if not self.options.branching_master:
            return None
    
        # 2. Cap on branches per node
        if self.branches_made >= self.options.max_branches:
            return None
    
        # 3. Legacy cutoff: if old_nbranch is on, never branch after branch_time_window
        if self.options.old_nbranch and self.age > self.options.branch_time_window:
            return None
    
        # 4. Secondary‐branch gate: only allow children if enabled
        if self.children and not self.options.secondary_branching:
            return None
    
        # 5. Hyphal‐density gate (crowding)
        if self.options.density_dependend:
            local_density = self.compute_local_hyphal_density(self.options.neighbour_radius)
            if local_density >= self.options.branching_density:
                return None
    
        # 6. Environmental field gate (max across samples)
        if self.field_aggregator:
            # decide which points to sample
            points = []
            if self.options.complete_evaluation:
                points = [end for _, end in self.subsegments]
            else:
                points = [self.end]
    
            if self.options.log_branch_points and len(self.subsegments) > 1:
                # also sample at the first branch junction
                points.append(self.subsegments[1][0])
    
            # compute max field strength
            strengths = []
            for pt in points:
                val, _ = self.field_aggregator.compute_field(pt, exclude_ids=[id(self)])
                strengths.append(val)
            if max(strengths) >= self.options.field_threshold:
                return None
    
        # 7. Random gate (branch_probability)
        if np.random.rand() >= self.options.branch_probability:
            return None
    
        # Passed all gates — create the branch
        angle = np.random.uniform(
            -self.options.branch_angle_spread,
             self.options.branch_angle_spread
        )
        axis = MPoint(0, 0, 1)
        rotated_orientation = self.orientation.copy().rotated_around(axis, angle)
    
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
    
        # 🧠 Memory‐based bias (unchanged)
        if self.options.direction_memory_blend > 0:
            rotated_orientation = (
                rotated_orientation.copy().scale(1 - self.options.direction_memory_blend)
                                  .add(self.direction_memory.copy().scale(self.options.direction_memory_blend))
                                  .normalise()
            )
    
        # 8. Leading‐branch selection
        keep_self_leading = (np.random.rand() < self.options.leading_branch_prob)
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
    
    def get_subsegments(self):
        return [(s.copy(), e.copy()) for s, e in self.subsegments]

    def __str__(self):
        status = "DEAD" if self.is_dead else ("TIP" if self.is_tip else "BRANCHED")
        return f"[{status}] {self.start} -> {self.end} | len={self.length:.2f}"
