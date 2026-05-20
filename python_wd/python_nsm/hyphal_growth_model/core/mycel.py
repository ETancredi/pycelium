# core/mycel.py

# Imports
from core.section import Section  # Mycelial segment
from core.point import MPoint     # 3D point/vector
from core.options import Options  # Simulation parameters container
from typing import Tuple          # Tuple type hint for RGB colour
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
        self.biomass_history: list[float] = []  # Total living biomass over time

    def seed(self, location: MPoint, orientation: MPoint, color: Tuple[float, float, float] = None):
        """Initialise the simulation with a single tip.
        Args:
            location: starting 3D point for seed segment
            orientation: direction vector for polarised growth
            color: optional RGB tuple for visualising mutations.
        """
        root = Section(
            start=location,             # Starting point of seed
            orientation=orientation,    # Growth direction vector
            opts=self.options,          # Pass simulation options into section
            parent=None,                # No parent (seed)
            color=color                 # Optional RGB color for visualising mutations
        )
        root.options = self.options         # Ensure section sees global options
        root.set_field_aggregator(None)     # Disable field aggregator until configured
        self.sections.append(root)          # Add seed to the section list

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
                "drug_mic": getattr(tip, "drug_mic", None), # MIC-like tolerance for this tip
                "drug_concentration": getattr(tip, "last_drug_concentration", None), # Last sampled drug concentration
                "drug_raw_growth_rate": getattr(tip, "last_drug_raw_growth_rate", None), # Signed Ψ from the drug-response model
                "drug_effective_growth_rate": getattr(tip, "last_drug_effective_growth_rate", None), # Non-negative rate actually applied
                "drug_growth_multiplier": getattr(tip, "last_drug_growth_multiplier", None) # Applied growth multiplier after clamping
            }
            for tip in self.get_tips()          # Iterate over active tips
        ]
        self.time_series.append(step_snapshot)

        # Advance simulation time
        self.time += self.options.time_step

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

        # If drug has pushed Ψ to zero or below, decide whether this tip simply
        # stalls in place or is removed from the active population.
        if raw_growth_rate <= 0.0:
            nonpositive_action = str(
                getattr(self.options, "drug_nonpositive_growth_action", "stall")
            ).strip().lower()

            if nonpositive_action in {"kill", "die", "death"}:
                section.is_dead = True
                section.is_tip = False
                section.drug_growth_stalled = False
                logger.debug(
                    "Drug kill: A=%.4g MIC=%.4g raw_growth=%.4g",
                    local_concentration,
                    mic,
                    raw_growth_rate,
                )

            elif nonpositive_action in {"stall", "pause", "clamp", "none"}:
                section.drug_growth_stalled = True
                logger.debug(
                    "Drug stall: A=%.4g MIC=%.4g raw_growth=%.4g",
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
