# core/section.py

# Imports:
import numpy as np # Numpy for random draws and vector math
import random # Python random for stochastic operations
import itertools # Creats a global unique ID generator
from core.point import MPoint # 3D point/vector class 
from core.options import Options # Simulation params container
from typing import Optional, Tuple # Optional and Tuple for type hints

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
        color: Optional[Tuple[float, float, float]] = None
    ):
        # Assign a unique integer ID to this Section
        self.id = next(_SECTION_ID_GEN)
        
        # Store starting point of segment (copy so original is not mutated)
        self.start = start.copy()
        
        # Store growth direction, normalised to unit length
        self.orientation = orientation.copy().normalise()
        
        # Initialise the physical length and age of segment
        self.length = 0.0
        self.age = 0.0

        # Flags to indicate if this segment is an active tip or has died
        self.is_tip = True
        self.is_dead = False

        # Counter for how many branches have emerged from this segment
        self.branches_made = 0

        # Link to parent Section (None for root) and list of child Sections
        self.parent = parent
        self.children = []

        # Current end point of this segment (starts equal to 0)
        self.end = self.start.copy()
        # List of individual sub-segments for detailed geometry tracking:
        # Each entry is a tuple (previous_point, new_point)
        self.subsegments = [(self.start.copy(), self.start.copy())]

        # Reference to global simulation options
        self.options = opts
        # Placeholder for a field aggregator (e.g. nutrient or density field)
        self.field_aggregator = None

        # Exponential moving-average of past directions for directional memory
        self.direction_memory = self.orientation.copy() 
        # Initialise RGB lineage color:
            # For seed segments, use passed-in color or fallback to opts.initial_color
            # for branches, colour must be passed by the caller
        if parent is None:
            # Use given colour or fallback to initial_colour from options
            self.color = color if color is not None else opts.initial_color
        else:
            self.color = color # Child branch colour is explicitly provided

    def set_field_aggregator(self, aggregator):
        """Assign a FieldAggregator for computing fields at this segment."""
        self.field_aggregator = aggregator

    def grow(self, rate: float, dt: float):
        """
        Grow the segment forward if it's active.
        Args:
            rate: base growth rate (length per time)
            dt: time increment
        """
        # Do nothing if this segment is not an active tip or is already dead
        if not self.is_tip or self.is_dead:
            return

        # If length-scaled growth is enabled, increase growth based on current length
        if self.options and self.options.length_scaled_growth:
            # Scale factor = 1+ length * coef
            scale_factor = 1 + self.length * self.options.length_growth_coef
            rate *= scale_factor

        # Compute how far to grow this time step
        growth_distance = rate * dt
        # Create a delta vector along orientation scaled by growth_distance
        delta = self.orientation.copy().scale(growth_distance)
        # Remember previous end for subsegment list
        prev_end = self.end.copy()
        # Move the end point by the delta vector
        self.end.add(delta)
        # Update this segment's accumulated length and age
        self.length += growth_distance
        self.age += dt
        # Record the subsegment from old end to new end
        self.subsegments.append((prev_end, self.end.copy()))

        # Volume Constraint Check (tip stops at boundary)
        opts = self.options
        if opts.volume_constraint:
            # Extract new coordinates
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
            # New memory = (1−α)*old_memory + α*current_orientation, then normalise
            self.direction_memory = (
                self.direction_memory.scale(1 - alpha)
                .add(self.orientation.copy().scale(alpha))
                .normalise()
            )

    def update(self):
        """Recompute length from start --> end and kill if segment is too small."""
        # Update length based on exact start-to-end distance
        self.length = self.start.distance_to(self.end)
        # If length is effectively 0, mark dead to avoid numerical issues
        if self.length < 1e-5:
            self.is_dead = True

    def maybe_branch(self, branch_chance: float, tip_count: int = 0) -> Optional["Section"]:
        """
        Decide whether to create a new branch at this tip.
        Returns:
            A new Section if branching occurs, otherwise None.
        """
        # Only active tips can branch
        if not self.is_tip or self.is_dead:
            return None
        #Respect maximum branches per segment
        if self.branches_made >= self.options.max_branches:
            return None
        # Enforce minimum age and length before branching
        if self.age < self.options.min_tip_age or self.length < self.options.min_tip_length:
            return None
        # Enforce maximum branching window by age
        if self.age > self.options.branch_time_window:
            return None
        # If a field aggregator exists, skip branching when field is too strong
        if self.field_aggregator:
            field_strength, _ = self.field_aggregator.compute_field(self.end, exclude_ids=[id(self)])
            if field_strength >= self.options.field_threshold:
                return None
        # Random chance to branch
        if np.random.rand() < branch_chance:
            # Pick a random rotation angle within allowed spread
            angle = np.random.uniform(-self.options.branch_angle_spread, self.options.branch_angle_spread)
            # Define Z-axis as rotation axis
            axis = MPoint(0, 0, 1)
            # Rotate current orientation around axis by angle
            rotated_orientation = self.orientation.copy().rotated_around(axis, angle)

            # Curvature bias
            if self.options.curvature_branch_bias > 0 and len(self.subsegments) >= 3:
                # Get last three subsegment endpoints to estimate curvatire
                p1 = self.subsegments[-3][0]
                p2 = self.subsegments[-2][0]
                p3 = self.subsegments[-1][1]
                # Compute unit direction vectors between points
                v1 = p2.copy().subtract(p1).normalise()
                v2 = p3.copy().subtract(p2).normalise()
                # Curvature vector = difference of consecutive direction vectors
                curve = v2.copy().subtract(v1).normalise()
                # Blend rotated orientation with curvature vector
                rotated_orientation = (
                    rotated_orientation.copy().scale(1.0 - self.options.curvature_branch_bias)
                    .add(curve.copy().scale(self.options.curvature_branch_bias))
                    .normalise()
                )
                print(f"🌀 Curvature blended into branch direction: strength={self.options.curvature_branch_bias:.2f}")

            # Directional memory-based bias
            if self.options.direction_memory_blend > 0:
                rotated_orientation = (
                    rotated_orientation.copy().scale(1.0 - self.options.direction_memory_blend)
                    .add(self.direction_memory.copy().scale(self.options.direction_memory_blend))
                    .normalise()
                )
                print(f"🧠 Directional memory blended into branch orientation: alpha={self.options.direction_memory_blend:.2f}")

            # Decide which branch retains "leading" growth (split vs. continue)
            keep_self_leading = np.random.rand() < self.options.leading_branch_prob
            if keep_self_leading:
                child_orientation = rotated_orientation
            else:
                # Swap orientations: parent keeps rotated, child keeps original
                child_orientation = self.orientation.copy()
                self.orientation = rotated_orientation

            # RGB Mutation color inheritance and Laplace distribution
            base_r, base_g, base_b = self.color
            new_r, new_g, new_b = base_r, base_g, base_b
            # If enabled, apply Laplace noise per colour channel with given probability
            if self.options.rgb_mutations_enabled and random.random() < self.options.color_mutation_prob:
                # Draw Laplace noise per channel
                dr = np.random.laplace(0.0, self.options.color_mutation_scale)
                dg = np.random.laplace(0.0, self.options.color_mutation_scale)
                db = np.random.laplace(0.0, self.options.color_mutation_scale)
                # Clamp mutated values back into [0,1]
                new_r = min(max(base_r + dr, 0.0), 1.0)
                new_g = min(max(base_g + dg, 0.0), 1.0)
                new_b = min(max(base_b + db, 0.0), 1.0)
            child_color = (new_r, new_g, new_b)

            # Instantiate the child Section
            child = Section(
                self.end.copy(), # Child starts at parent's end
                child_orientation, # Direction for new branch
                opts=self.options, # Inherit global options
                parent=self, # Link back to parent
                color=child_color # Assign possibly mutated colour
            )
            # Child is by definition a tip
            child.is_tip = True
            # Inherit directional memory from parent
            child.direction_memory = self.direction_memory.copy()
            # Share the same field aggregator so fields remain consistent
            child.set_field_aggregator(self.field_aggregator)

            # Record new child as a direct branch of this segment
            self.children.append(child)
            # Increment this segment's branch count
            self.branches_made += 1
            return child # Return new branch

        return None # If not, no branching occurs

    def get_subsegments(self):
        """
        Return deep copies of all stores subsegment pairs.
        Each entry is a tuple (start_point, end_point).
        """
        return [(s.copy(), e.copy()) for s, e in self.subsegments]

    def __str__(self):
        """
        Readable string: shows status (TIP/BRANCHED/DEAD),
        Start and end coordinates, and current length.
        """
        status = "DEAD" if self.is_dead else ("TIP" if self.is_tip else "BRANCHED")
        return f"[{status}] {self.start} -> {self.end} | len={self.length:.2f}"
