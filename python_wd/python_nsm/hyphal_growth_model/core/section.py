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

        if self.branches_made >= self.options.max_branches:
            return None

        if (self.options.enforce_min_tip_age and self.age < self.options.min_tip_age) \
           or (self.options.enforce_min_tip_length and self.length < self.options.min_tip_length):
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

        def get_new_growing_vector(self, default_strength: float):
        parent = self.orientation.copy().normalise()
        max_angle_rad = math.radians(self.options.branch_angle_spread)

        # 1) Optimal orientation via field (unchanged)
        if self.options.optimal_branch_orientation and self.field_aggregator:
            strength, field_vec = self.field_aggregator.compute_field(
                self.end, exclude_ids=[id(self)]
            )

            if field_vec.length() > 0:
                return field_vec.copy().normalise().scale(default_strength)

        # 2) Sample a random vector within the cone around `parent`
        #    by sampling uniformly on the spherical cap:
        #    cos(phi) in [cos(max_angle), 1], azimuth theta in [0, 2π)
        cos_max = math.cos(max_angle_rad)
        u = np.random.uniform(cos_max, 1.0)
        phi = math.acos(u)                     # polar angle from parent_dir
        theta = np.random.uniform(0, 2 * math.pi)
    
        # Build an orthonormal basis (parent, perp1, perp2):
        # Find a vector not parallel to parent
        if abs(parent.coords[0]) < 0.9:
            temp = MPoint(1,0,0)
        else:
            temp = MPoint(0,1,0)
        perp1 = parent.cross(temp).normalise()   # first perpendicular
        perp2 = parent.cross(perp1).normalise()  # second perpendicular
    
        # Spherical‐cap direction:
        # new_dir = cos(phi)*parent + sin(phi)*(cos(theta)*perp1 + sin(theta)*perp2)
        part1 = parent.copy().scale(math.cos(phi))
        part2 = perp1.copy().scale(math.sin(phi) * math.cos(theta))
        part3 = perp2.copy().scale(math.sin(phi) * math.sin(theta))
        new_dir = part1.add(part2).add(part3).normalise().scale(default_strength)
    
        return new_dir
    
    def get_subsegments(self):
        return [(s.copy(), e.copy()) for s, e in self.subsegments]

    def __str__(self):
        status = "DEAD" if self.is_dead else ("TIP" if self.is_tip else "BRANCHED")
        return f"[{status}] {self.start} -> {self.end} | len={self.length:.2f}"
