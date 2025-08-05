# core/mycel.py

# Imports
from core.section import Section # Mycelial segment
from core.point import MPoint # 3D point/vector
from core.options import Options # Simulation parameters container
from typing import Tuple # Tuple tupe hint for RGB colour
import numpy as np # Random choice and numerical ops

class Mycel:
    """Main simulation engine: manages sections and steps simulation forward."""

    def __init__(self, options: Options):
        self.sections: list[Section] = [] # All sections in the simulation
        self.options = options # Simulation parameters
        self.time = 0.0 # Current simulation time
        self.step_history = [] # History of tip positions per step
        self.time_series = [] # Snapshot of tip data at each step
        self.biomass_history: list[float] = [] # Total living biomass over time

    def seed(self, location: MPoint, orientation: MPoint, color: Tuple[float, float, float] = None):
        """Initialise the simulation with a single tip.
        Args:
            location: starting 3D point for seed segment
            orientation: direction vector for polarise growth
            colour: optional RGB tuple for visualising mutations.
        """
        root = Section(
            start=location, # Starting point of seed
            orientation=orientation, # Growth direction vector
            opts=self.options, # Pass simulation options into section
            parent=None, # No parent (seed)
            color=color # Optional RGB color for visualising mutations
        )
        root.options = self.options # Ensure section sees global options
        root.set_field_aggregator(None) # Disable field aggregator until configured
        self.sections.append(root) # Add seed to the section list

    def step(self):
        """Advance the simulation by one time step:
        1. Grow existing segments
        2. Apply destructor checks
        3. Attempt branching for tips or internal nodes
        4. Record data snapshots
        5. Prune excess tips if needed
        6. Update histories and increment time.
        """
        new_sections = [] # Hold branches created this step

        # Log step start
        print(f"\n🔄 STEP START: {self.time:.2f}") # Log beginning of step
        print(f"  Total sections: {len(self.sections)}") # Log current total no. sections

        tip_count = len(self.get_tips()) # Count active tips before growth

        # 1. Grow & update existing sections
        for section in self.sections:
            if section.is_dead: # Skip dead segments
                continue

            # Grow by growth_rate over time_step
            section.grow(self.options.growth_rate, self.options.time_step) # Increase section length based on growth_rate and time_step
            section.update() # Update internal state (e.g. age increment, orientation adjustments)

            # Log active tips
            if section.is_tip and not section.is_dead: # If section remains alive, log end point, length and age
                print(f"    🌱 TIP @ {section.end} (len={section.length:.2f}, age={section.age:.2f})")

        # 2. Destructor logic: prine tips based on age, length, density. nutrient, isolation
        for section in self.sections:
            # Only consdier alive tips for destruction
            if not section.is_tip or section.is_dead:
                continue

            # A) Die if exceeding max age
            if self.options.die_if_old and section.age > self.options.max_age:
                section.is_dead = True
                continue

            # B) Die if exceeding max length
            if section.length > self.options.max_length:
                section.is_dead = True
                continue

            # C) Density kill if too crowded and using field aggregator
            if self.options.die_if_too_dense and section.field_aggregator:
                # Compute scalar field (e.g. crowding) at section end
                density = section.field_aggregator.compute_field(section.end)[0]
                # Kill if density exceeds threshold
                if density > self.options.density_threshold:
                    print(f"💀 Density kill: {density:.3f} > {self.options.density_threshold:.3f}")
                    section.is_dead = True
                    continue

            # D) Nutrient repulsion kill
            if self.options.use_nutrient_field and section.field_aggregator:
                # Compute nutrient field at section end
                nutrient_field = section.field_aggregator.compute_field(section.end)[0]
                # Kill if nutrient field is too repellent (negative beyond threshold)
                if nutrient_field < -abs(self.options.nutrient_repulsion):
                    print(f"💀 Repellent kill: nutrient field too negative ({nutrient_field:.3f})")
                    section.is_dead = True
                    continue

            # E) Isolation kill if too few neighbours within radius
            if section.field_aggregator:
                nearby_count = 0 # Counter fr tips within neighbourhood radius
                for other in self.get_tips():
                    if other is section: # Skip self-comparison
                        continue
                    # Increment if another tip is within radius
                    if section.end.distance_to(other.end) <= self.options.neighbour_radius:
                        nearby_count += 1

                # Kill if isolation threshold not met
                if nearby_count < self.options.min_supported_tips:
                    print(f"🧊 Killed due to isolation: {nearby_count} < {self.options.min_supported_tips}")
                    section.is_dead = True
                    continue
                    
        # Log state of last-checked section (could be any tip)
        print(f"🛠️ After destructor: Tip {section.end} | is_tip={section.is_tip} | is_dead={section.is_dead}")
        # Useful to confirm flags on a representative segment

        # 3. Branching: attempt to create new sections from tips or internal nodes
        for section in self.sections:
            if section.is_dead:
                continue # Skip dead segments

            # Allow branching if it's a tip, or if internal branching is enabled
            if section.is_tip or self.options.allow_internal_branching:
                # maybe_branch returns a new Section if branching occurs
                child = section.maybe_branch(self.options.branch_probability, tip_count=tip_count)
                
                if child: # If branching succeeded
                    print(f"    🌿 BRANCHED: {section.end} → {child.orientation}")
                    new_sections.append(child) # Queue the new section for addition

        # Add newly created sections to the master list
        self.sections.extend(new_sections)
        
        # 4. Record a snapshot of current tip data (positions and metrics)
        step_snapshot = [
            {
                "time": self.time, # Current simulation time
                "x": tip.end.coords[0], # X-coord
                "y": tip.end.coords[1], # Y-coord
                "z": tip.end.coords[2], # Z-coord
                "age": tip.age, # Age of tip segment
                "length": tip.length # Length of tip segment
            }
            for tip in self.get_tips() # Iterate over active tips
        ]
        self.time_series.append(step_snapshot) # Append snapshot to time_series
        
        # Advance simulation time
        self.time += self.options.time_step

        # 5. Optional pruning: limit total active tips if above max_supproted_tips
        if hasattr(self.options, "max_supported_tips") and self.options.max_supported_tips > 0:
            active_tips = self.get_tips() # Recompute list of active tips
            # If count exceeds allowed maximum
            if len(active_tips) > self.options.max_supported_tips:
                print(f"⚠️ Tip pruning: {len(active_tips)} tips exceed max ({self.options.max_supported_tips})")

                # Compute how many to remove
                excess = len(active_tips) - self.options.max_supported_tips
                # Randomly select tips to kill without replacement
                to_prune = np.random.choice(active_tips, size=excess, replace=False)

                for tip in to_prune:
                    tip.is_dead = True # Mark selected tip(s) as dead
                    print(f"💀 Pruned tip at {tip.end} due to overcrowding")

        # Log summary of new sections added this step
        print(f"  📦 Added {len(new_sections)} new sections.")
        for s in new_sections:
            print(f"    ➕ New tip: is_tip={s.is_tip}, is_dead={s.is_dead}, orientation={s.orientation}")

        # Re-count active tips after branching and pruning
        tip_count = len(self.get_tips())
        print(f"  🔚 STEP END: {tip_count} active tips")
        
        # 6. Update history and biomass tracking
        tip_data = [(tip.end.coords[0], tip.end.coords[1], tip.end.coords[2]) for tip in self.get_tips()]
        self.step_history.append((self.time, tip_data)) # Append position history
        # Compute total living biomass (sum of lengths of all non-dead sections)
        total_biomass = sum(sec.length for sec in self.sections if not sec.is_dead)
        self.biomass_history.append(total_biomass) # Record biomass
        print(f" 🪵 Total living biomass: {total_biomass:.2f}")

    def get_tips(self):
        """Return list of sections that are tips and not dead."""
        return [s for s in self.sections if s.is_tip and not s.is_dead]

    def get_all_segments(self):
        """Return all sections, reguardless of status."""
        return self.sections

    def __str__(self):
        """Summary of current simulation state."""
        return f"Mycel @ t={self.time:.2f} | tips={len(self.get_tips())} | total={len(self.sections)}"
