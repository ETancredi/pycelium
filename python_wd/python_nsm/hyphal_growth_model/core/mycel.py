# core/mycel.py

from core.section import Section
from core.point import MPoint
from core.options import Options
import numpy as np

class Mycel:
    """Main simulation engine: manages sections and steps simulation forward."""

    def __init__(self, options: Options):
        self.sections: list[Section] = []
        self.options = options
        self.time = 0.0
        self.step_history = []
        self.time_series = []
        self.biomass_history: list[float] = []

    def seed(self, location: MPoint, orientation: MPoint):
        """Initialise the simulation with a single tip."""
        root = Section(start=location, orientation=orientation)
        root.options = self.options
        root.set_field_aggregator(None)  
        self.sections.append(root)

    def step(self):
        """Advance the simulation by one time step."""
        new_sections = []

        print(f"\n🔄 STEP START: {self.time:.2f}")
        print(f"  Total sections: {len(self.sections)}")

        tip_count = len(self.get_tips())

        # Grow & update existing sections
        for section in self.sections:
            if section.is_dead:
                continue

            section.grow(self.options.growth_rate, self.options.time_step)
            section.update()

            if section.is_tip and not section.is_dead:
                print(f"    🌱 TIP @ {section.end} (len={section.length:.2f}, age={section.age:.2f})")

        # Run destructor logic: age, density, nutrient, and neighborhood checks
        for section in self.sections:
            if not section.is_tip or section.is_dead:
                continue

            if self.options.die_if_old and section.age > self.options.max_age:
                section.is_dead = True
                continue

            if section.length > self.options.max_length:
                section.is_dead = True
                continue

            if self.options.die_if_too_dense and section.field_aggregator:
                density = section.field_aggregator.compute_field(section.end)[0]
                if density > self.options.density_threshold:
                    print(f"💀 Density kill: {density:.3f} > {self.options.density_threshold:.3f}")
                    section.is_dead = True
                    continue

            if self.options.use_nutrient_field and section.field_aggregator:
                nutrient_field = section.field_aggregator.compute_field(section.end)[0]
                if nutrient_field < -abs(self.options.nutrient_repulsion):
                    print(f"💀 Repellent kill: nutrient field too negative ({nutrient_field:.3f})")
                    section.is_dead = True
                    continue

            if section.field_aggregator:
                nearby_count = 0
                for other in self.get_tips():
                    if other is section:
                        continue
                    if section.end.distance_to(other.end) <= self.options.neighbour_radius:
                        nearby_count += 1

                if nearby_count < self.options.min_supported_tips:
                    print(f"🧊 Killed due to isolation: {nearby_count} < {self.options.min_supported_tips}")
                    section.is_dead = True
                    continue
        
        print(f"🛠️ After destructor: Tip {section.end} | is_tip={section.is_tip} | is_dead={section.is_dead}")

        # Try branching
        for section in self.sections:
            if section.is_dead:
                continue

            if section.is_tip or self.options.allow_internal_branching:
                child = section.maybe_branch(self.options.branch_probability, tip_count=tip_count)
                if child:
                    print(f"    🌿 BRANCHED: {section.end} → {child.orientation}")
                    new_sections.append(child)

        self.sections.extend(new_sections)
        
        # Snapshot tip positions
        step_snapshot = [
            {
                "time": self.time,
                "x": tip.end.coords[0],
                "y": tip.end.coords[1],
                "z": tip.end.coords[2],
                "age": tip.age,
                "length": tip.length
            }
            for tip in self.get_tips()
        ]
        self.time_series.append(step_snapshot)
        self.time += self.options.time_step

        # NEW: max_supported_tips pruning logic
        if hasattr(self.options, "max_supported_tips") and self.options.max_supported_tips > 0:
            active_tips = self.get_tips()
            if len(active_tips) > self.options.max_supported_tips:
                print(f"⚠️ Tip pruning: {len(active_tips)} tips exceed max ({self.options.max_supported_tips})")

                excess = len(active_tips) - self.options.max_supported_tips
                to_prune = np.random.choice(active_tips, size=excess, replace=False)

                for tip in to_prune:
                    tip.is_dead = True
                    print(f"💀 Pruned tip at {tip.end} due to overcrowding")

        print(f"  📦 Added {len(new_sections)} new sections.")
        for s in new_sections:
            print(f"    ➕ New tip: is_tip={s.is_tip}, is_dead={s.is_dead}, orientation={s.orientation}")

        tip_count = len(self.get_tips())
        print(f"  🔚 STEP END: {tip_count} active tips")
        
        # Log positions of all active tips at this step
        tip_data = [(tip.end.coords[0], tip.end.coords[1], tip.end.coords[2]) for tip in self.get_tips()]
        self.step_history.append((self.time, tip_data))

        # Compoute & record total "biomass" this step
        total_biomass = sum(sec.length for sec in self.sections if not sec.is_dead)
        self.biomass_history.append(total_biomass)
        print(f" 🪵 Total living biomass: {total_biomass:.2f}")

    def get_tips(self):
        return [s for s in self.sections if s.is_tip and not s.is_dead]

    def get_all_segments(self):
        return self.sections

    def __str__(self):
        return f"Mycel @ t={self.time:.2f} | tips={len(self.get_tips())} | total={len(self.sections)}"
