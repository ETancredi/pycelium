# core/section.py

# Imports:
import numpy as np # Numpy for random draws and vector math
import random # Python random for stochastic operations
import itertools # Creats a global unique ID generator
import colorsys # Deterministic palette generation for mutant MIC lineages
import logging # for silencing prints and converting to debug logs
logger = logging.getLogger("pycelium.core.section")
from core.point import MPoint # 3D point/vector class 
from core.options import Options # Simulation params container
from typing import Optional, Tuple # Optional and Tuple for type hints

# Global counter for unique Section IDs
_SECTION_ID_GEN = itertools.count()
# Global counter for distinct MIC-mutant lineage colours
_MIC_MUTATION_LINEAGE_ID_GEN = itertools.count(1)

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

        # Germ-tube lineage metadata.
        # Normal lateral branches inherit the germ_tube_order of their parent.
        # Delayed secondary germ tubes are created explicitly by Mycel and then
        # relabelled as "secondary_germ_tube" so they can be filtered later.
        if parent is None:
            self.germ_tube_order = 0
            self.germ_tube_role = "primary_germ_tube"
            self.emergence_step = 0
            self.emergence_time = 0.0
        else:
            self.germ_tube_order = getattr(parent, "germ_tube_order", 0)
            self.germ_tube_role = "branch"
            self.emergence_step = getattr(parent, "emergence_step", None)
            self.emergence_time = getattr(parent, "emergence_time", None)

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

        # MIC-mutation lineage colouring metadata.  When enabled, every new MIC
        # mutant lineage receives its own inherited colour so breakouts can be
        # recognised across PNGs, CSVs, OBJ/HTML exports, and MP4 animations.
        if parent is None:
            self.mic_mutation_lineage_id = 0
            self.mic_mutation_origin_section_id = None
            self.visual_color_source = "wildtype"
        else:
            self.mic_mutation_lineage_id = getattr(parent, "mic_mutation_lineage_id", 0)
            self.mic_mutation_origin_section_id = getattr(parent, "mic_mutation_origin_section_id", None)
            self.visual_color_source = getattr(parent, "visual_color_source", "inherited")

        # Initialise visible section colour.
        # - With MIC-lineage colouring enabled, non-mutants inherit their parent
        #   colour and wildtype/root sections use a shared wildtype colour.
        # - Otherwise fall back to the older RGB mutation visualisation logic.
        if getattr(opts, "drug_mic_lineage_coloring_enabled", False):
            if parent is None:
                self.color = color if color is not None else getattr(opts, "drug_mic_wildtype_color", (0.1, 0.1, 0.1))
            else:
                self.color = color if color is not None else getattr(parent, "color", getattr(opts, "drug_mic_wildtype_color", (0.1, 0.1, 0.1)))
        else:
            # Initialise RGB lineage color:
            # For seed segments, use passed-in color or fallback to opts.initial_color
            # for branches, colour must be passed by the caller
            if parent is None:
                # Use given colour or fallback to initial_colour from options
                self.color = color if color is not None else opts.initial_color
            else:
                self.color = color # Child branch colour is explicitly provided

        # Antifungal tolerance carried by this section.
        # At this stage it is simply inherited from the parent or seeded from
        # opts.drug_wildtype_mic. In the next mutation pass, this value can be
        # changed in daughter branches to model MIC-like resistance evolution.
        if parent is None:
            self.drug_mic = getattr(opts, "drug_wildtype_mic", 1.0)
        else:
            self.drug_mic = getattr(parent, "drug_mic", getattr(opts, "drug_wildtype_mic", 1.0))

        # MIC mutation lineage metadata.  These fields make it possible to audit
        # whether a breakout lineage was created by a tolerance mutation, and by
        # how much its MIC changed relative to its immediate parent.
        self.drug_mic_parent = (
            getattr(parent, "drug_mic", None)
            if parent is not None
            else None
        )
        self.drug_mic_mutated_from_parent = False
        self.drug_mic_mutated_from_wildtype = (
            abs(self.drug_mic - getattr(opts, "drug_wildtype_mic", self.drug_mic)) > 1e-12
        )
        self.drug_mic_mutation_delta_log = 0.0
        self.drug_mic_mutation_probability = 0.0
        # Antifungal response diagnostics. These are refreshed during Mycel.step()
        # whenever a drug field is active, then exported to CSV for analysis.
        self.last_drug_concentration = 0.0 # Local antifungal concentration last sampled by this tip
        self.last_drug_growth_multiplier = 1.0 # Applied growth multiplier after clamping non-positive rates
        self.last_drug_raw_growth_rate = getattr(opts, "growth_rate", 1.0) # Signed Ψ from the response model
        self.last_drug_effective_growth_rate = getattr(opts, "growth_rate", 1.0) # Non-negative rate passed to grow()
        self.drug_growth_stalled = False # True when drug prevented extension this step without killing the tip
        self.drug_growth_stopped_by_drug = False # Persistent flag: this hypha has reached zero/negative drug-adjusted growth or been killed by drug
        self.drug_growth_stopped_time = None # Simulation time when drug first stopped or killed this hypha, filled by Mycel.step()
        self.drug_growth_stop_reason = "" # stall or kill, for visual/CSV diagnostics
        self.drug_killed_by_drug = False # Persistent flag: this tip was killed by a fungicidal drug response
        self.drug_death_time = None # Simulation time when fungicidal drug killing first occurred

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

    def _clamp_probability(self, value: float) -> float:
        """Return a probability constrained to the interval [0, 1]."""
        return min(max(float(value), 0.0), 1.0)

    def _mic_mutant_palette_color(self, lineage_id: int) -> Tuple[float, float, float]:
        """Generate a reproducible high-contrast colour for one MIC-mutant lineage."""
        # Golden-angle hue spacing spreads successive lineage colours well around
        # the colour wheel and gives us an effectively unbounded palette.
        hue = (0.618033988749895 * max(int(lineage_id), 1)) % 1.0
        saturation = min(max(float(getattr(self.options, "drug_mic_palette_saturation", 0.8)), 0.0), 1.0)
        value = min(max(float(getattr(self.options, "drug_mic_palette_value", 0.95)), 0.0), 1.0)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return (float(r), float(g), float(b))


    def _drug_mic_mutation_probability(self) -> float:
        """
        Calculate the per-new-section MIC mutation probability.

        Density is not used directly here.  Instead, dense internal colony
        regions create more mutation opportunities indirectly because they keep
        producing more new hyphal sections/branches while outer wildtype tips
        are drug-stalled.
        """
        opts = self.options
        if not getattr(opts, "drug_mic_mutations_enabled", False):
            return 0.0

        base_probability = self._clamp_probability(
            getattr(opts, "drug_mic_mutation_base_probability", 0.0)
        )
        max_probability = self._clamp_probability(
            getattr(opts, "drug_mic_mutation_max_probability", 1.0)
        )
        return min(base_probability, max_probability)

    def apply_drug_mic_mutation_from_parent(self, parent: "Section") -> None:
        """
        Possibly mutate this new section's MIC relative to its parent.

        The mutation effect is multiplicative, implemented on the log scale:

            new_MIC = parent_MIC * exp(delta)
            delta ~ Laplace(0, drug_mic_mutation_scale)

        A positive delta makes the daughter more resistant; a negative delta makes
        it more susceptible.  This gives many small changes and exponentially
        fewer large jumps in either direction.
        """
        opts = self.options
        parent_mic = float(getattr(parent, "drug_mic", getattr(opts, "drug_wildtype_mic", 1.0)))
        wildtype_mic = float(getattr(opts, "drug_wildtype_mic", parent_mic))
        use_mic_lineage_colours = bool(getattr(opts, "drug_mic_lineage_coloring_enabled", False))

        mutation_probability = parent._drug_mic_mutation_probability()

        # Record attempted-mutation context even when no mutation occurs.
        self.drug_mic_parent = parent_mic
        self.drug_mic = parent_mic
        self.drug_mic_mutation_probability = mutation_probability
        self.drug_mic_mutation_delta_log = 0.0
        self.drug_mic_mutated_from_parent = False
        self.drug_mic_mutated_from_wildtype = abs(parent_mic - wildtype_mic) > 1e-12

        # Unless a fresh MIC mutation occurs, the daughter stays in the same MIC
        # lineage and inherits the same visual identity as its parent.
        self.mic_mutation_lineage_id = getattr(parent, "mic_mutation_lineage_id", 0)
        self.mic_mutation_origin_section_id = getattr(parent, "mic_mutation_origin_section_id", None)
        self.visual_color_source = getattr(parent, "visual_color_source", "wildtype")
        if use_mic_lineage_colours:
            self.color = getattr(parent, "color", getattr(opts, "drug_mic_wildtype_color", (0.1, 0.1, 0.1)))

        if mutation_probability <= 0.0:
            return

        if np.random.rand() >= mutation_probability:
            return

        scale = max(0.0, float(getattr(opts, "drug_mic_mutation_scale", 0.25)))
        delta_log = float(np.random.laplace(0.0, scale)) if scale > 0.0 else 0.0

        # Clamp the resulting MIC to positive, finite bounds so rare very large
        # Laplace draws cannot destabilise long parameter sweeps.
        min_mic = max(float(getattr(opts, "drug_mic_min", 1e-6)), 1e-300)
        max_mic = max(float(getattr(opts, "drug_mic_max", 1e6)), min_mic)
        mutated_mic = parent_mic * float(np.exp(delta_log))
        mutated_mic = min(max(mutated_mic, min_mic), max_mic)

        self.drug_mic = mutated_mic
        self.drug_mic_mutation_delta_log = delta_log
        self.drug_mic_mutated_from_parent = abs(mutated_mic - parent_mic) > 1e-12
        self.drug_mic_mutated_from_wildtype = abs(mutated_mic - wildtype_mic) > 1e-12

        if self.drug_mic_mutated_from_parent:
            self.mic_mutation_lineage_id = next(_MIC_MUTATION_LINEAGE_ID_GEN)
            self.mic_mutation_origin_section_id = self.id
            self.visual_color_source = "mic_mutant"
            if use_mic_lineage_colours:
                self.color = self._mic_mutant_palette_color(self.mic_mutation_lineage_id)

        logger.debug(
            "MIC mutation: parent_MIC=%.6g child_MIC=%.6g delta_log=%.6g p=%.6g",
            parent_mic,
            mutated_mic,
            delta_log,
            mutation_probability,
        )

    def maybe_branch(self, branch_chance: float, tip_count: int = 0) -> Optional["Section"]:
        """
        Decide whether to create a new branch at this tip.
        Returns:
            A new Section if branching occurs, otherwise None.
        """
        # Only active tips can branch
        if not self.is_tip or self.is_dead:
            return None
        # Respect maximum branches per segment
        if self.branches_made >= self.options.max_branches:
            return None
        # Enforce minimum age and length before branching
        if self.age < self.options.min_tip_age or self.length < self.options.min_tip_length:
            return None
        # Enforce maximum branching window by age
        if self.age > self.options.branch_time_window:
            return None
        # If a field aggregator exists, skip branching when field is too strong.
        # This remains ordinary crowding/field inhibition, not MIC-mutation boosting.
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
                logger.debug("Curvature blended into branch direction: strength=%s", locals().get("curv_strength", locals().get("field_strength", "n/a")))

            # Directional memory-based bias
            if self.options.direction_memory_blend > 0:
                rotated_orientation = (
                    rotated_orientation.copy().scale(1.0 - self.options.direction_memory_blend)
                    .add(self.direction_memory.copy().scale(self.options.direction_memory_blend))
                    .normalise()
                )
                logger.debug("Directional memory blended into branch orientation: alpha=%s", locals().get("alpha", "n/a"))

            # Decide which branch retains "leading" growth (split vs. continue)
            keep_self_leading = np.random.rand() < self.options.leading_branch_prob
            if keep_self_leading:
                child_orientation = rotated_orientation
            else:
                # Swap orientations: parent keeps rotated, child keeps original
                child_orientation = self.orientation.copy()
                self.orientation = rotated_orientation

            # Visible colour inheritance.  When MIC-lineage colouring is enabled,
            # ordinary branches keep the parent lineage colour and only true MIC
            # mutation events create a new colour.  Otherwise retain the older
            # optional RGB-mutation visualisation.
            if getattr(self.options, "drug_mic_lineage_coloring_enabled", False):
                child_color = self.color
            else:
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
            # Apply optional MIC mutation after the daughter exists so all audit
            # metadata are stored directly on the new Section.
            child.apply_drug_mic_mutation_from_parent(parent=self)
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
