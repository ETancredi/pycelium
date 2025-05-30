# core/mycel.py

from __future__ import annotations
import numpy as np
from core.point import MPoint
from core.section import Section
from core.options import Options, ToggleableFloat, ToggleableInt

class Mycel:
    """Main simulation engine: manages sections and steps simulation forward."""

    def __init__(self, options: Options):
        self.sections: list[Section] = []
        self.options = options
        self.time = 0.0
        self.step_history = []
        self.time_series = []

    def seed(self, location: MPoint, orientation: MPoint):
        """Initialise the simulation with a single tip."""
        root = Section(start=location, orientation=orientation)
        root.options = self.options
        root.set_field_aggregator(None)
        root.set_orientator(None)
        self.sections.append(root)

    def step(self):
        """Advance the simulation by one time step."""
        new_sections: list[Section] = []

        print(f"\n🔄 STEP START: {self.time:.2f}")
        print(f"  Total sections: {len(self.sections)}")

        tip_count = len(self.get_tips())

        # 1) Grow & update
        for section in self.sections:
            if section.is_dead:
                continue

            section.grow(self.options.growth_rate, self.options.time_step)
            section.update()

            if section.is_tip and not section.is_dead:
                print(f"    🌱 TIP @ {section.end} (len={section.length:.2f}, age={section.age:.2f})")

        # 2) Destructor logic: age, length, density, nutrient, neighborhood
        for section in self.sections:
            if not section.is_tip or section.is_dead:
                continue

            # — death by age —
            if self.options.die_if_old:
                ma = self.options.max_age
                if isinstance(ma, ToggleableFloat):
                    if ma.enabled and section.age > ma.value:
                        section.is_dead = True
                        continue
                else:
                    if section.age > ma:
                        section.is_dead = True
                        continue

            # — death by length —
            ml = self.options.max_length
            if isinstance(ml, ToggleableFloat):
                if ml.enabled and section.length > ml.value:
                    section.is_dead = True
                    continue
            else:
                if section.length > ml:
                    section.is_dead = True
                    continue

            # — death by density field (exclude self) —
            if self.options.die_if_too_dense and section.field_aggregator:
                density = section.field_aggregator.compute_field(
                    section.end,
                    exclude_ids=[id(section)]
                )[0]
                dt = self.options.density_threshold
                if isinstance(dt, ToggleableFloat):
                    thr = dt.value if dt.enabled else float('inf')
                else:
                    thr = dt
                if density > thr:
                    print(f"💀 Density kill: {density:.3f} > {thr:.3f}")
                    section.is_dead = True
                    continue

            # — death by nutrient repellent —
            if self.options.use_nutrient_field and section.field_aggregator:
                nf = section.field_aggregator.compute_field(section.end)[0]
                if nf < -abs(self.options.nutrient_repulsion):
                    print(f"💀 Repellent kill: nutrient field too negative ({nf:.3f})")
                    section.is_dead = True
                    continue

            # — neighborhood support check —
            if section.field_aggregator:
                # unwrap neighbour_radius
                nr = self.options.neighbour_radius
                rad = nr.value if isinstance(nr, ToggleableFloat) and nr.enabled else (nr if not isinstance(nr, ToggleableFloat) else 0.0)

                nearby_count = 0
                for other in self.get_tips():
                    if other is section:
                        continue
                    if section.end.distance_to(other.end) <= rad:
                        nearby_count += 1

                # min_supported_tips
                mst = self.options.min_supported_tips
                if isinstance(mst, ToggleableInt):
                    if mst.enabled and nearby_count < mst.value:
                        print(f"🧊 Killed due to isolation: {nearby_count} < {mst.value}")
                        section.is_dead = True
                        continue
                else:
                    if nearby_count < mst:
                        print(f"🧊 Killed due to isolation: {nearby_count} < {mst}")
                        section.is_dead = True
                        continue

                # max_supported_tips
                mxt = self.options.max_supported_tips
                if isinstance(mxt, ToggleableInt):
                    if mxt.enabled and nearby_count > mxt.value:
                        print(f"⚠️ Killed due to overcrowding: {nearby_count} > {mxt.value}")
                        section.is_dead = True
                        continue
                else:
                    if nearby_count > mxt:
                        print(f"⚠️ Killed due to overcrowding: {nearby_count} > {mxt}")
                        section.is_dead = True
                        continue

            print(f"🛠️ After destructor: Tip {section.end} | is_tip={section.is_tip} | is_dead={section.is_dead}")

        # 3) Branching
        for section in self.sections:
            if section.is_dead:
                continue

            if section.is_tip or self.options.allow_internal_branching:
                child = section.maybe_branch(self.options.branch_probability, tip_count=tip_count)
                if child:
                    print(f"    🌿 BRANCHED: {section.end} → {child.orientation}")
                    new_sections.append(child)

        self.sections.extend(new_sections)

        # 4) Record tip snapshot
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

        # 5) Tip‐count pruning
        mxt = self.options.max_supported_tips
        limit = None
        if isinstance(mxt, ToggleableInt) and mxt.enabled:
            limit = mxt.value
        elif not isinstance(mxt, ToggleableInt) and mxt > 0:
            limit = mxt

        if limit is not None:
            active_tips = self.get_tips()
            if len(active_tips) > limit:
                print(f"⚠️ Tip pruning: {len(active_tips)} tips exceed max ({limit})")
                excess = len(active_tips) - limit
                to_prune = np.random.choice(active_tips, size=excess, replace=False)
                for tip in to_prune:
                    tip.is_dead = True
                    print(f"💀 Pruned tip at {tip.end} due to overcrowding")

        print(f"  📦 Added {len(new_sections)} new sections.")
        for s in new_sections:
            print(f"    ➕ New tip: is_tip={s.is_tip}, is_dead={s.is_dead}, orientation={s.orientation}")

        tip_count = len(self.get_tips())
        print(f"  🔚 STEP END: {tip_count} active tips")
        
        # 6) Log positions of all active tips at this step
        tip_data = [(tip.end.coords[0], tip.end.coords[1], tip.end.coords[2]) for tip in self.get_tips()]
        self.step_history.append((self.time, tip_data))

    def get_tips(self) -> list[Section]:
        return [s for s in self.sections if s.is_tip and not s.is_dead]

    def get_all_segments(self) -> list[Section]:
        return self.sections

    def __str__(self) -> str:
        return f"Mycel @ t={self.time:.2f} | tips={len(self.get_tips())} | total={len(self.sections)}"
