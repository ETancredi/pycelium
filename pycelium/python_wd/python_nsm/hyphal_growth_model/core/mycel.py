# core/mycel.py

# Imports
from core.section import Section  # Mycelial segment
from core.point import MPoint     # 3D point/vector
from core.options import Options  # Simulation parameters container
from typing import Tuple, List     # Tuple/List type hints
import numpy as np                # Random choice and numerical ops
import logging # Logging (quiet by default; control with PYCELIUM_LOG_LEVEL)
logger = logging.getLogger("pycelium.core.mycel")

class Mycel:
    """Main simulation engine: manages sections and steps simulation forward."""

    def __init__(self, options: Options):
        self.sections: list[Section] = []  # All sections in the simulation
        self.options = options             # Simulation parameters
        self.time = 0.0                    # Current simulation time
        self.step_history = []             # History of tip positions per step
        self.time_series = []              # Snapshot of tip data at each step
        self.network_time_series = []      # Snapshot of full 2D segment geometry at each step for line-based MP4s
        self.biomass_history: list[float] = []  # Total living biomass over time
        self.step_index = 0                # Completed simulation steps; step_index + 1 is the current 1-indexed step

        # Germination state for delayed secondary germ tubes.
        self.germination_origin: MPoint | None = None
        self.primary_germ_tube: Section | None = None
        self.primary_germ_orientation: MPoint | None = None
        self.secondary_germ_tube_schedule: List[int] = []
        self.secondary_germ_tubes_emitted = 0

    def seed(self, location: MPoint, orientation: MPoint, color: Tuple[float, float, float] = None):
        """Initialise the simulation with a single tip.
        Args:
            location: starting 3D point for seed segment
            orientation: direction vector for polarised growth
            color: optional RGB tuple for visualising mutations.
        """
        if getattr(self.options, "drug_mic_lineage_coloring_enabled", False):
            seed_color = getattr(self.options, "drug_mic_wildtype_color", (0.1, 0.1, 0.1))
        else:
            seed_color = color

        root = Section(
            start=location,             # Starting point of seed
            orientation=orientation,    # Growth direction vector
            opts=self.options,          # Pass simulation options into section
            parent=None,                # No parent (seed)
            color=seed_color            # Optional RGB color for visualising mutations
        )
        root.options = self.options         # Ensure section sees global options
        root.set_field_aggregator(None)     # Disable field aggregator until configured
        self.sections.append(root)          # Add seed to the section list

        # Store the original spore position and the primary germ-tube direction.
        # Delayed secondary germ tubes use these to emerge from the same origin,
        # usually in the opposite direction from the first germ tube.
        self.germination_origin = location.copy()
        self.primary_germ_tube = root
        self.primary_germ_orientation = orientation.copy().normalise()
        self.secondary_germ_tube_schedule = self._build_secondary_germ_tube_schedule()
        self.secondary_germ_tubes_emitted = 0

        if self.secondary_germ_tube_schedule:
            logger.info(
                "Secondary germ tube schedule: %s",
                ", ".join(str(x) for x in self.secondary_germ_tube_schedule),
            )

    def _build_secondary_germ_tube_schedule(self) -> List[int]:
        """
        Decide how many delayed germ tubes will emerge and at which steps.

        This is intentionally scheduled once at seeding time rather than checked
        independently every step. That gives a clean biological interpretation:
        a spore has an intrinsic germination programme, with emergence times drawn
        from a distribution that becomes more likely later in the lag window.
        """
        opts = self.options
        if not getattr(opts, "secondary_germ_tubes_enabled", False):
            return []

        min_step = max(1, int(getattr(opts, "secondary_germ_tube_min_step", 2)))
        max_step = max(min_step, int(getattr(opts, "secondary_germ_tube_max_step", 10)))

        min_count = max(0, int(getattr(opts, "secondary_germ_tube_min_count", 1)))
        max_count = max(min_count, int(getattr(opts, "secondary_germ_tube_max_count", 4)))

        count = min_count
        p_extra = float(getattr(opts, "secondary_germ_tube_extra_probability", 0.35))
        p_extra = min(max(p_extra, 0.0), 1.0)
        decay = max(0.0, float(getattr(opts, "secondary_germ_tube_extra_probability_decay", 0.5)))

        # Geometric-like tail: one delayed germ tube is most likely, two is less
        # likely, three less likely again, etc.
        while count < max_count and np.random.rand() < p_extra:
            count += 1
            p_extra = min(max(p_extra * decay, 0.0), 1.0)

        if count <= 0:
            return []

        possible_steps = np.arange(min_step, max_step + 1, dtype=int)
        shape = max(0.0, float(getattr(opts, "secondary_germ_tube_timing_shape", 1.0)))
        weights = (possible_steps - min_step + 1).astype(float) ** shape
        weights = weights / weights.sum()

        # Prefer distinct emergence steps when possible. If the requested maximum
        # count exceeds the number of available steps, replacement is allowed.
        replace = count > len(possible_steps)
        scheduled = np.random.choice(possible_steps, size=count, replace=replace, p=weights)
        return sorted(int(x) for x in scheduled)

    def _secondary_germ_tube_orientation(self) -> MPoint:
        """Return a new orientation for a delayed germ tube."""
        opts = self.options
        if self.primary_germ_orientation is None:
            base = MPoint(1.0, 0.0, 0.0)
        else:
            base = self.primary_germ_orientation.copy()

        mean_angle = float(getattr(opts, "secondary_germ_tube_angle_degrees", 180.0))
        spread = max(0.0, float(getattr(opts, "secondary_germ_tube_angle_spread", 25.0)))
        angle = mean_angle + np.random.uniform(-spread, spread)

        # In the current 2D model, the natural way to make an opposite germ tube
        # is to rotate in the x-y plane. This also works safely in 3D, although it
        # only jitters around the z-axis rather than sampling a full 3D cone.
        orientation = base.rotated_around(MPoint(0, 0, 1), angle).normalise()

        if getattr(opts, "use_2d", False):
            orientation.coords[2] = 0.0
            orientation.normalise()

        return orientation

    def _create_secondary_germ_tube(self, scheduled_step: int) -> Section | None:
        """Create one delayed germ tube from the original spore position."""
        if self.germination_origin is None or self.primary_germ_tube is None:
            return None

        order = self.secondary_germ_tubes_emitted + 1
        tube = Section(
            start=self.germination_origin.copy(),
            orientation=self._secondary_germ_tube_orientation(),
            opts=self.options,
            parent=self.primary_germ_tube,
            color=getattr(self.primary_germ_tube, "color", self.options.initial_color),
        )

        # Override the default child metadata from Section.__init__ so downstream
        # CSVs can distinguish true delayed germ tubes from ordinary branches.
        tube.germ_tube_order = order
        tube.germ_tube_role = "secondary_germ_tube"
        tube.emergence_step = scheduled_step
        tube.emergence_time = self.time
        tube.set_field_aggregator(getattr(self.primary_germ_tube, "field_aggregator", None))

        # Secondary germ tubes are also new hyphal lineages, so they may carry an
        # MIC mutation if that model is enabled.
        tube.apply_drug_mic_mutation_from_parent(parent=self.primary_germ_tube)

        # Keep a graph connection to the primary germ tube for lineage/network
        # traversal, but do not increment branches_made: this event is not a
        # normal lateral branch and should not consume the branch budget.
        self.primary_germ_tube.children.append(tube)
        self.secondary_germ_tubes_emitted += 1
        logger.info(
            "Secondary germ tube emerged: order=%d step=%d orientation=%s",
            order,
            scheduled_step,
            tube.orientation,
        )
        return tube

    def _emit_due_secondary_germ_tubes(self) -> List[Section]:
        """Create all delayed germ tubes whose scheduled step has arrived."""
        if not self.secondary_germ_tube_schedule:
            return []

        current_step = self.step_index + 1  # user-facing step number; first Mycel.step() is step 1
        new_tubes = []
        while self.secondary_germ_tube_schedule and self.secondary_germ_tube_schedule[0] <= current_step:
            scheduled_step = self.secondary_germ_tube_schedule.pop(0)
            tube = self._create_secondary_germ_tube(scheduled_step)
            if tube is not None:
                new_tubes.append(tube)
        return new_tubes

    def _last_visible_subsegment_xy(self, section):
        """Return the final non-zero x/y subsegment for a section, if present."""
        for start, end in reversed(getattr(section, "subsegments", [])):
            x0, y0 = float(start.coords[0]), float(start.coords[1])
            x1, y1 = float(end.coords[0]), float(end.coords[1])
            if abs(x1 - x0) >= 1e-12 or abs(y1 - y0) >= 1e-12:
                return ((x0, y0), (x1, y1))

        x0, y0 = float(section.start.coords[0]), float(section.start.coords[1])
        x1, y1 = float(section.end.coords[0]), float(section.end.coords[1])
        if abs(x1 - x0) >= 1e-12 or abs(y1 - y0) >= 1e-12:
            return ((x0, y0), (x1, y1))
        return None

    def _record_network_snapshot(self) -> None:
        """Record the current visible 2D network geometry for animation.

        The older MP4 path only stored tip end-points, which made the colony look
        like a cloud of dots.  For a biologically useful movie we also need the
        hyphal links that were present at each step.  This stores lightweight
        x/y coordinate pairs for every non-zero subsegment plus the current live
        tip positions.  It is intentionally in-memory only; the final video is
        written at the end of the run.
        """
        # Only keep these snapshots when an MP4 is requested.  Large parameter
        # sweeps can turn generate_mycelium_growth_mp4 off to avoid storing
        # per-frame geometry in memory.
        if not getattr(self.options, "generate_mycelium_growth_mp4", False):
            return

        line_segments = []
        line_colors = []
        for section in self.sections:
            section_color = tuple(getattr(section, "color", (0.0, 0.0, 0.0)))
            for start, end in section.subsegments:
                x0, y0 = float(start.coords[0]), float(start.coords[1])
                x1, y1 = float(end.coords[0]), float(end.coords[1])

                # Ignore the initial zero-length placeholder and any later
                # zero-length stalled entries.
                if abs(x1 - x0) < 1e-12 and abs(y1 - y0) < 1e-12:
                    continue

                line_segments.append(((x0, y0), (x1, y1)))
                line_colors.append(section_color)

        live_tips = []
        tip_colors = []
        for tip in self.get_tips():
            live_tips.append((float(tip.end.coords[0]), float(tip.end.coords[1])))
            tip_colors.append(tuple(getattr(tip, "color", (0.0, 0.0, 0.0))))

        stopped_segments = []
        stopped_tips = []
        for section in self.sections:
            if getattr(section, "drug_growth_stopped_by_drug", False):
                final_segment = self._last_visible_subsegment_xy(section)
                if final_segment is not None:
                    stopped_segments.append(final_segment)
                stopped_tips.append((float(section.end.coords[0]), float(section.end.coords[1])))

        self.network_time_series.append({
            "time": float(self.time),
            "segments": line_segments,
            "segment_colors": line_colors,
            "tips": live_tips,
            "tip_colors": tip_colors,
            "stopped_segments": stopped_segments,
            "stopped_tips": stopped_tips,
        })

    def step(self, drug_field=None):
        """Advance the simulation by one time step:
        1. Grow existing segments
        2. Apply destructor checks
        3. Attempt branching for tips or internal nodes
        4. Record data snapshots
        5. Prune excess tips if needed
        6. Update histories and increment time.

        Args:
            drug_field:
                Optional DrugField2D instance. When supplied, each active tip
                samples local antifungal concentration and its growth rate is
                reduced according to the configured MIC / Hill-response model.
        """
        new_sections = []  # Hold branches created this step

        # Step start (debug-only)
        logger.debug("STEP START: t=%.2f | total_sections=%d", self.time, len(self.sections))

        tip_count = len(self.get_tips())  # Count active tips before growth

        # 1) Grow & update existing sections
        for section in self.sections:
            if section.is_dead:  # Skip dead segments
                continue

            # Compute the effective local growth rate for this section.
            # If no drug field is present, this simply returns options.growth_rate.
            # If drug is enabled, the tip samples local concentration and receives
            # a pharmacodynamic growth rate before Section.grow() applies its
            # existing length-scaled growth logic.
            effective_growth_rate = self._drug_adjusted_growth_rate(section, drug_field)

            # The MIC-centred pharmacodynamic model can return zero or negative
            # raw growth rates. Those rates should not move hyphae backwards.
            # _drug_adjusted_growth_rate() therefore clamps the applied rate to
            # zero and sets section.drug_growth_stalled when drug has halted this
            # tip without killing it.
            if section.is_dead:
                continue
            if getattr(section, "drug_growth_stalled", False):
                # A stalled tip remains in place. Age is still incremented so the
                # record reflects elapsed exposure time, but Section.update() is
                # deliberately skipped because it treats zero-length seed tips as
                # dead numerical artefacts.
                section.age += self.options.time_step
                continue

            # Grow by effective_growth_rate over time_step.
            section.grow(effective_growth_rate, self.options.time_step)
            section.update()  # Update internal state (e.g. age increment, orientation adjustments)

            # Debug trace for living tips
            if section.is_tip and not section.is_dead:
                logger.debug("TIP pos=%s len=%.2f age=%.2f", section.end, section.length, section.age)

        # 2) Destructor logic: prune tips based on age, length, density, nutrient, isolation
        for section in self.sections:
            # Only consider alive tips for destruction
            if not section.is_tip or section.is_dead:
                continue

            # A) Die if exceeding max age
            if self.options.die_if_old and section.age > self.options.max_age:
                section.is_dead = True
                logger.debug("Tip died of age: age=%.2f > max_age=%.2f", section.age, self.options.max_age)
                continue

            # B) Die if exceeding max length
            if section.length > self.options.max_length:
                section.is_dead = True
                logger.debug("Tip died of length: len=%.2f > max_len=%.2f", section.length, self.options.max_length)
                continue

            # C) Density kill if too crowded and using field aggregator
            if self.options.die_if_too_dense and section.field_aggregator:
                # Compute scalar field (e.g. crowding) at section end
                density = section.field_aggregator.compute_field(section.end)[0]
                if density > self.options.density_threshold:
                    logger.debug("Density kill: %.3f > threshold %.3f", density, self.options.density_threshold)
                    section.is_dead = True
                    continue

            # D) Nutrient repulsion kill
            if self.options.use_nutrient_field and section.field_aggregator:
                nutrient_field = section.field_aggregator.compute_field(section.end)[0]
                # Kill if nutrient field is too repellent (negative beyond threshold)
                if nutrient_field < -abs(self.options.nutrient_repulsion):
                    logger.debug("Repellent kill: nutrient_field=%.3f < -|repulsion|=%.3f",
                                 nutrient_field, abs(self.options.nutrient_repulsion))
                    section.is_dead = True
                    continue

            # E) Isolation kill if too few neighbours within radius
            if section.field_aggregator:
                nearby_count = 0  # Counter for tips within neighbourhood radius
                for other in self.get_tips():
                    if other is section:
                        continue
                    if section.end.distance_to(other.end) <= self.options.neighbour_radius:
                        nearby_count += 1

                if nearby_count < self.options.min_supported_tips:
                    logger.debug("Isolation kill: neighbours=%d < min_supported=%d",
                                 nearby_count, self.options.min_supported_tips)
                    section.is_dead = True
                    continue

        # Log state of last-checked section (representative)
        # (safe even if loop didn’t run; only logs when 'section' exists and is a tip)
        try:
            logger.debug("Post-destruction sample tip: pos=%s is_tip=%s is_dead=%s",
                         section.end, section.is_tip, section.is_dead)
        except UnboundLocalError:
            # No tips iterated; ignore
            pass

        # 3) Branching: attempt to create new sections from tips or internal nodes
        for section in self.sections:
            if section.is_dead:
                continue  # Skip dead segments

            # Allow branching if it's a tip, or if internal branching is enabled
            if section.is_tip or self.options.allow_internal_branching:
                # maybe_branch returns a new Section if branching occurs
                child = section.maybe_branch(self.options.branch_probability, tip_count=tip_count)

                if child:  # If branching succeeded
                    logger.debug("BRANCHED: %s → %s", section.end, child.orientation)
                    new_sections.append(child)  # Queue the new section for addition

        # Add delayed secondary germ tubes from the original spore when their
        # scheduled lag time has elapsed. These are separate from normal branches:
        # they begin at the germination origin and usually point opposite to the
        # primary germ tube.
        new_sections.extend(self._emit_due_secondary_germ_tubes())

        # Add newly created sections to the master list
        if new_sections:
            logger.debug("Added %d new sections this step", len(new_sections))
        self.sections.extend(new_sections)

        # 4) Record a snapshot of current tip data (positions and metrics)
        step_snapshot = [
            {
                "time": self.time,              # Current simulation time
                "x": tip.end.coords[0],         # X-coord
                "y": tip.end.coords[1],         # Y-coord
                "z": tip.end.coords[2],         # Z-coord
                "age": tip.age,                 # Age of tip segment
                "length": tip.length,           # Length of tip segment
                "germ_tube_order": getattr(tip, "germ_tube_order", None), # Germ-tube lineage index; 0 is the primary germ tube
                "germ_tube_role": getattr(tip, "germ_tube_role", None), # primary_germ_tube, secondary_germ_tube, or branch
                "emergence_step": getattr(tip, "emergence_step", None), # Step at which this germ-tube lineage emerged
                "emergence_time": getattr(tip, "emergence_time", None), # Simulation time at emergence
                "drug_mic": getattr(tip, "drug_mic", None), # MIC-like tolerance for this tip
                "drug_mic_parent": getattr(tip, "drug_mic_parent", None), # Immediate-parent MIC inherited before mutation
                "drug_mic_mutated_from_parent": getattr(tip, "drug_mic_mutated_from_parent", None), # True if this tip's MIC changed on birth
                "drug_mic_mutated_from_wildtype": getattr(tip, "drug_mic_mutated_from_wildtype", None), # True if MIC differs from wildtype baseline
                "drug_mic_mutation_delta_log": getattr(tip, "drug_mic_mutation_delta_log", None), # log(new_MIC / parent_MIC)
                "drug_mic_mutation_probability": getattr(tip, "drug_mic_mutation_probability", None), # Effective probability used at birth
                "mic_mutation_lineage_id": getattr(tip, "mic_mutation_lineage_id", None), # Inherited colour lineage ID for MIC-mutant breakout tracking
                "mic_mutation_origin_section_id": getattr(tip, "mic_mutation_origin_section_id", None), # Section ID where the current mutant lineage began
                "visual_color_source": getattr(tip, "visual_color_source", None), # Whether colour reflects wildtype inheritance or a MIC-mutation lineage
                "r": getattr(tip, "color", (None, None, None))[0], # Red channel for this tip
                "g": getattr(tip, "color", (None, None, None))[1], # Green channel for this tip
                "b": getattr(tip, "color", (None, None, None))[2], # Blue channel for this tip
                "drug_concentration": getattr(tip, "last_drug_concentration", None), # Last sampled drug concentration
                "drug_raw_growth_rate": getattr(tip, "last_drug_raw_growth_rate", None), # Signed Ψ from the drug-response model
                "drug_effective_growth_rate": getattr(tip, "last_drug_effective_growth_rate", None), # Non-negative rate actually applied
                "drug_growth_multiplier": getattr(tip, "last_drug_growth_multiplier", None), # Applied growth multiplier after clamping
                "drug_growth_stopped_by_drug": getattr(tip, "drug_growth_stopped_by_drug", None), # Persistent drug-stopped marker
                "drug_growth_stopped_time": getattr(tip, "drug_growth_stopped_time", None), # First time drug stopped this hypha
                "drug_growth_stop_reason": getattr(tip, "drug_growth_stop_reason", None), # stall or kill
                "drug_killed_by_drug": getattr(tip, "drug_killed_by_drug", None), # Persistent fungicidal-death marker
                "drug_death_time": getattr(tip, "drug_death_time", None) # First time drug killed this hypha
            }
            for tip in self.get_tips()          # Iterate over active tips
        ]
        self.time_series.append(step_snapshot)

        # Record line-based geometry for the upgraded 2D MP4.
        self._record_network_snapshot()

        # Advance simulation time and completed-step counter
        self.time += self.options.time_step
        self.step_index += 1

        # 5) Optional pruning: limit total active tips if above max_supported_tips
        if hasattr(self.options, "max_supported_tips") and self.options.max_supported_tips > 0:
            active_tips = self.get_tips()  # Recompute list of active tips
            if len(active_tips) > self.options.max_supported_tips:
                # One informative line at INFO (can be changed to DEBUG if desired)
                logger.info(
                    "Tip pruning: %d tips exceed max (%d) → pruning",
                    len(active_tips), self.options.max_supported_tips
                )

                excess = len(active_tips) - self.options.max_supported_tips
                to_prune = np.random.choice(active_tips, size=excess, replace=False)

                for tip in to_prune:
                    tip.is_dead = True
                    logger.debug("Pruned tip at %s due to overcrowding", tip.end)

        # 6) Update history and biomass tracking
        tip_data = [(tip.end.coords[0], tip.end.coords[1], tip.end.coords[2]) for tip in self.get_tips()]
        self.step_history.append((self.time, tip_data))

        # Compute total living biomass (sum of lengths of all non-dead sections)
        total_biomass = sum(sec.length for sec in self.sections if not sec.is_dead)
        self.biomass_history.append(total_biomass)
        logger.debug("STEP END: active_tips=%d | biomass=%.2f", len(self.get_tips()), total_biomass)

    def _drug_adjusted_growth_rate(self, section: Section, drug_field=None) -> float:
        """
        Return the local growth rate after antifungal response.

        In the current MIC-calibrated model, local concentration is converted into
        an absolute pharmacodynamic growth rate Ψ. This is different from the old
        multiplier-only curve: when A == MIC, Ψ is exactly zero. Concentrations
        above MIC produce negative raw Ψ values, which are then handled according
        to options.drug_nonpositive_growth_action.
        """
        # Ψmax is the normal drug-free growth rate already used by Pycelium.
        max_growth_rate = self.options.growth_rate

        # Non-tip or dead sections cannot grow; Section.grow() will also guard
        # this, but returning the base rate avoids unnecessary drug sampling.
        if not section.is_tip or section.is_dead:
            return max_growth_rate

        # If drug is disabled, keep explicit diagnostic attributes so downstream
        # CSV exports have consistent columns.
        if drug_field is None:
            section.last_drug_concentration = 0.0
            section.last_drug_raw_growth_rate = max_growth_rate
            section.last_drug_effective_growth_rate = max_growth_rate
            section.last_drug_growth_multiplier = 1.0
            section.drug_growth_stalled = False
            if not getattr(section, "drug_growth_stopped_by_drug", False):
                section.drug_growth_stop_reason = ""
            return max_growth_rate

        # Sample local drug concentration at the current growing tip endpoint.
        local_concentration = drug_field.sample(section.end)

        # MIC is inherited per Section and seeded from opts.drug_wildtype_mic.
        # Later mutation logic can alter section.drug_mic in newly produced branches.
        mic = getattr(section, "drug_mic", getattr(self.options, "drug_wildtype_mic", 1.0))

        # Select the response model. The pharmacodynamic model is the new default
        # because it gives the desired MIC behaviour: A == MIC -> zero growth.
        response_model = str(getattr(self.options, "drug_response_model", "pharmacodynamic")).strip().lower()

        if response_model in {"pharmacodynamic", "mic", "mic_pharmacodynamic"}:
            # Compute signed Ψ from the MIC-centred pharmacodynamic equation.
            raw_growth_rate = drug_field.pharmacodynamic_growth_rate(
                concentration=local_concentration,
                mic=mic,
                max_growth_rate=max_growth_rate,
                min_growth_rate=getattr(self.options, "drug_min_growth_rate", -max_growth_rate),
                hill_coefficient=getattr(self.options, "drug_hill_coefficient", 4.0),
            )

        elif response_model in {"hill_multiplier", "legacy", "legacy_hill"}:
            # Backward-compatible path: local concentration gives a multiplier,
            # not a true MIC-calibrated growth-rate crossing.
            multiplier = drug_field.growth_multiplier(
                concentration=local_concentration,
                mic=mic,
                hill_coefficient=getattr(self.options, "drug_hill_coefficient", 4.0),
                min_multiplier=getattr(self.options, "drug_min_growth_multiplier", 0.0),
            )
            raw_growth_rate = max_growth_rate * multiplier

        else:
            raise ValueError(
                "Unknown drug_response_model: "
                f"{response_model!r}. Use 'pharmacodynamic' or 'hill_multiplier'."
            )

        # Raw Ψ can be negative when local drug is above MIC. Negative extension
        # would move the hypha backwards, which is not meaningful in this growth
        # model, so the applied rate is clamped to zero.
        effective_growth_rate = max(raw_growth_rate, 0.0)

        # Store diagnostics before any kill/stall decision so final CSVs still
        # explain why a tip stopped.
        section.last_drug_concentration = local_concentration
        section.last_drug_raw_growth_rate = raw_growth_rate
        section.last_drug_effective_growth_rate = effective_growth_rate
        section.last_drug_growth_multiplier = (
            effective_growth_rate / max_growth_rate if max_growth_rate > 0.0 else 0.0
        )
        section.drug_growth_stalled = False

        # Resolve the high-level drug effect mode.  New configs should prefer
        # drug_effect_type (fungistatic vs. fungicidal); legacy configs can keep
        # using drug_nonpositive_growth_action.
        effect_type = str(getattr(self.options, "drug_effect_type", "legacy")).strip().lower()
        if effect_type in {"fungistatic", "static", "stall", "stalled"}:
            nonpositive_action = "stall"
        elif effect_type in {"fungicidal", "cidal", "kill", "deadly"}:
            nonpositive_action = "kill"
        elif effect_type in {"legacy", "auto", "inherit", "compat", "compatibility", ""}:
            nonpositive_action = str(
                getattr(self.options, "drug_nonpositive_growth_action", "stall")
            ).strip().lower()
        else:
            raise ValueError(
                "Unknown drug_effect_type: "
                f"{effect_type!r}. Use 'fungistatic', 'fungicidal', or 'legacy'."
            )

        kill_triggered = False
        kill_reason = ""
        if nonpositive_action in {"kill", "die", "death"}:
            fungicidal_condition = str(
                getattr(self.options, "drug_fungicidal_condition", "raw_growth_nonpositive")
            ).strip().lower()

            if fungicidal_condition in {"raw_growth_nonpositive", "raw_nonpositive", "nonpositive_growth", "raw_growth_le_zero"}:
                kill_triggered = raw_growth_rate <= 0.0
                kill_reason = "kill_raw_growth_nonpositive"
            elif fungicidal_condition in {"concentration_at_or_above_mic", "mic_exceeded", "concentration_ge_mic", "at_or_above_mic", "local_concentration_ge_mic"}:
                kill_triggered = local_concentration >= mic
                kill_reason = "kill_concentration_at_or_above_mic"
            elif fungicidal_condition in {"either", "raw_or_mic_exceeded", "raw_or_concentration_at_or_above_mic"}:
                kill_triggered = (raw_growth_rate <= 0.0) or (local_concentration >= mic)
                kill_reason = "kill_raw_or_mic_threshold"
            else:
                raise ValueError(
                    "Unknown drug_fungicidal_condition: "
                    f"{fungicidal_condition!r}. Use 'raw_growth_nonpositive' or 'concentration_at_or_above_mic'."
                )

        growth_stopped = (raw_growth_rate <= 0.0) or kill_triggered
        if growth_stopped and not getattr(section, "drug_growth_stopped_by_drug", False):
            section.drug_growth_stopped_by_drug = True
            section.drug_growth_stopped_time = self.time

        # If drug has pushed Ψ to zero or below, or if fungicidal logic says the
        # local concentration itself is lethal, decide whether this tip simply
        # stalls in place or is removed from the active population.
        if kill_triggered:
            section.is_dead = True
            section.is_tip = False
            section.drug_growth_stalled = False
            section.drug_growth_stop_reason = kill_reason or "kill"
            if not getattr(section, "drug_killed_by_drug", False):
                section.drug_killed_by_drug = True
                section.drug_death_time = self.time
            logger.debug(
                "Drug kill: A=%.4g MIC=%.4g raw_growth=%.4g reason=%s",
                local_concentration,
                mic,
                raw_growth_rate,
                section.drug_growth_stop_reason,
            )

        elif raw_growth_rate <= 0.0:
            if nonpositive_action in {"stall", "pause", "clamp", "none"}:
                section.drug_growth_stalled = True
                section.drug_growth_stop_reason = "stall"
                logger.debug(
                    "Drug stall: A=%.4g MIC=%.4g raw_growth=%.4g",
                    local_concentration,
                    mic,
                    raw_growth_rate,
                )

            elif nonpositive_action in {"kill", "die", "death"}:
                section.is_dead = True
                section.is_tip = False
                section.drug_growth_stalled = False
                section.drug_growth_stop_reason = "kill_raw_growth_nonpositive"
                if not getattr(section, "drug_killed_by_drug", False):
                    section.drug_killed_by_drug = True
                    section.drug_death_time = self.time
                logger.debug(
                    "Drug kill: A=%.4g MIC=%.4g raw_growth=%.4g",
                    local_concentration,
                    mic,
                    raw_growth_rate,
                )

            else:
                raise ValueError(
                    "Unknown drug_nonpositive_growth_action: "
                    f"{nonpositive_action!r}. Use 'stall' or 'kill'."
                )

        # Return the non-negative rate that may be passed to Section.grow().
        return effective_growth_rate

    def get_tips(self):
        """Return list of sections that are tips and not dead."""
        return [s for s in self.sections if s.is_tip and not s.is_dead]

    def get_all_segments(self):
        """Return all sections, regardless of status."""
        return self.sections

    def __str__(self):
        """Summary of current simulation state."""
        return f"Mycel @ t={self.time:.2f} | tips={len(self.get_tips())} | total={len(self.sections)}"
