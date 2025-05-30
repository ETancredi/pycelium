# core/section.py

from __future__ import annotations
import math
import numpy as np
from core.point import MPoint
from core.options import ToggleableFloat, ToggleableInt, Options
from tropisms.sect_field_finder import SectFieldFinder


class Section:
    """Represents a single hyphal segment (tip or branch) in the fungal network"""

    def __init__(self, start: MPoint, orientation: MPoint, parent: Section | None = None):
        self.start = start.copy()
        self.orientation = orientation.copy().normalise()
        self.length = 0.0
        self.age = 0.0

        self.is_tip = True
        self.is_dead = False
        self.branches_made = 0

        self.parent = parent
        self.children: list[Section] = []

        self.end = self.start.copy()
        self.subsegments: list[tuple[MPoint, MPoint]] = [(self.start.copy(), self.start.copy())]

        self.options = None
        self.field_aggregator = None
        self.orientator = None    # Shared Orientator instance

        self.direction_memory = self.orientation.copy()

    def set_field_aggregator(self, aggregator):
        self.field_aggregator = aggregator

    def set_orientator(self, orientator):
        """Assign the shared Orientator for tropism blending."""
        self.orientator = orientator

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

        # 1. Tropism/orientator‐driven re‐orientation
        if self.orientator:
            self.orientation = self.orientator.compute(self)

        # 2. Length‐scaled growth (unwrap ToggleableFloat)
        if self.options.length_scaled_growth:
            lgc = self.options.length_growth_coef
            coef = (
                lgc.value if isinstance(lgc, ToggleableFloat) and lgc.enabled
                else (lgc if not isinstance(lgc, ToggleableFloat) else 0.0)
            )
            rate *= (1 + self.length * coef)

        # 3. Age‐based slowdown (unwrap ToggleableFloat)
        age = self.age if self.age > 0 else 1.0
        dag = self.options.d_age
        exp = (
            dag.value if isinstance(dag, ToggleableFloat) and dag.enabled
            else (dag if not isinstance(dag, ToggleableFloat) else 0.0)
        )
        rate /= (age ** exp)

        # 4. Default growth‐vector scaling (unwrap ToggleableFloat)
        dgv = self.options.default_growth_vector
        if isinstance(dgv, ToggleableFloat):
            if dgv.enabled:
                rate *= dgv.value
        else:
            rate *= dgv

        # 5. Actual growth step
        growth_distance = rate * dt
        delta = self.orientation.copy().scale(growth_distance)

        prev_end = self.end.copy()
        self.end.add(delta)
        self.length += growth_distance
        self.age += dt
        self.subsegments.append((prev_end, self.end.copy()))

        # 6. EMA‐style directional memory update (unwrap ToggleableFloat)
        dmb = self.options.direction_memory_blend
        alpha = (
            dmb.value if isinstance(dmb, ToggleableFloat) and dmb.enabled
            else (dmb if not isinstance(dmb, ToggleableFloat) else 0.0)
        )
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

        # Only live tips can branch
        if not self.is_tip or self.is_dead:
            print("DEBUG: gate=not_a_live_tip")
            return None

        # 1. Master switch
        if not self.options.branching_master:
            print("DEBUG: gate=branching_master OFF")
            return None
        print("DEBUG: passed branching_master")
        
        # 2. Cap on branches per node
        mb = self.options.max_branches
        maxb = mb.value if isinstance(mb, ToggleableInt) and mb.enabled else (
               mb if not isinstance(mb, ToggleableInt) else 0)
        if self.branches_made >= maxb:
            print(f"DEBUG: gate=max_branches (made={self.branches_made} ≥ {maxb})")
            return None
        print(f"DEBUG: passed max_branches ({self.branches_made} < {maxb})")

        # 3. Legacy cutoff
        if self.options.old_nbranch and self.age > self.options.branch_time_window:
            print(f"DEBUG: gate=old_nbranch (age={self.age:.2f} > {self.options.branch_time_window})")
            return None
        print("DEBUG: passed old_nbranch")

        # 4. Secondary‐branch gate
        if self.children and not self.options.secondary_branching:
            print("DEBUG: gate=secondary_branching OFF on non-first branch")
            return None
        print("DEBUG: passed secondary_branching")

        # 5. Crowding gate
        if self.options.density_dependend:
            # unwrap neighbour_radius
            nr = self.options.neighbour_radius
            radius = nr.value if isinstance(nr, ToggleableFloat) and nr.enabled else (
                     nr if not isinstance(nr, ToggleableFloat) else 0.0)

            local_density = self.compute_local_hyphal_density(radius)

            # unwrap branching_density
            bd = self.options.branching_density
            thr_bd = bd.value if isinstance(bd, ToggleableFloat) and bd.enabled else (
                     bd if not isinstance(bd, ToggleableFloat) else 0.0)

            if local_density >= thr_bd:
                print(f"DEBUG: gate=density_dependend (density={local_density:.3f} ≥ {thr_bd})")
                return None
        print("DEBUG: passed density_dependend")

        # 6. Environmental‐field gate
        if self.field_aggregator:
            # choose sample points
            points = (
                [end for _, end in self.subsegments]
                if self.options.complete_evaluation else [self.end]
            )
            if self.options.log_branch_points and len(self.subsegments) > 1:
                points.append(self.subsegments[1][0])

            strengths = [
                self.field_aggregator.compute_field(pt, exclude_ids=[id(self)])[0]
                for pt in points
            ]

            # unwrap field_threshold
            ft = self.options.field_threshold
            thr_ft = ft.value if isinstance(ft, ToggleableFloat) and ft.enabled else (
                     ft if not isinstance(ft, ToggleableFloat) else float("inf"))

            if max(strengths) >= thr_ft:
                print(f"DEBUG: gate=environmental_field (max_field={max(strengths):.3f} ≥ {thr_ft})")
                return None
        print("DEBUG: passed environmental_field")

        # 7. Random gate
        bp = self.options.branch_probability
        prob = bp.value if isinstance(bp, ToggleableFloat) and bp.enabled else (
               bp if not isinstance(bp, ToggleableFloat) else 0.0)
        if np.random.rand() >= prob:
            print(f"DEBUG: gate=random (rand={r:.3f} ≥ {prob})")
            return None
        print(f"DEBUG: passed random (rand={r:.3f} < {prob})")

        # ALL GATES PASSED --> BRANCH
        print("DBG: all gates passed, creating branch")

        # 8. Compute branch orientation
        dgv = self.options.default_growth_vector
        default_strength = dgv.value if isinstance(dgv, ToggleableFloat) and dgv.enabled else (
                           dgv if not isinstance(dgv, ToggleableFloat) else 1.0)
        rotated_orientation = self.get_new_growing_vector(default_strength)

        # 🌀 Curvature bias
        cb = self.options.curvature_branch_bias
        bias = cb.value if isinstance(cb, ToggleableFloat) and cb.enabled else (
               cb if not isinstance(cb, ToggleableFloat) else 0.0)
        if bias > 0 and len(self.subsegments) >= 3:
            p1 = self.subsegments[-3][0]
            p2 = self.subsegments[-2][0]
            p3 = self.subsegments[-1][1]
            v1 = p2.copy().subtract(p1).normalise()
            v2 = p3.copy().subtract(p2).normalise()
            curve = v2.copy().subtract(v1).normalise()
            rotated_orientation = (
                rotated_orientation.copy()
                                   .scale(1 - bias)
                                   .add(curve.copy().scale(bias))
                                   .normalise()
            )

        # 🧠 Directional‐memory bias
        dmb = self.options.direction_memory_blend
        blend = dmb.value if isinstance(dmb, ToggleableFloat) and dmb.enabled else (
                dmb if not isinstance(dmb, ToggleableFloat) else 0.0)
        if blend > 0:
            rotated_orientation = (
                rotated_orientation.copy().scale(1 - blend)
                                   .add(self.direction_memory.copy().scale(blend))
                                   .normalise()
            )

        # 9. Leading‐branch selection
        lbp = self.options.leading_branch_prob
        leadp = lbp.value if isinstance(lbp, ToggleableFloat) and lbp.enabled else (
                lbp if not isinstance(lbp, ToggleableFloat) else 0.0)
        if np.random.rand() < leadp:
            child_orient = rotated_orientation
        else:
            child_orient = self.orientation.copy()
            self.orientation = rotated_orientation

        # create & return the new tip
        child = Section(self.end.copy(), child_orient, parent=self)
        child.is_tip = True
        child.options = self.options
        child.direction_memory = self.direction_memory.copy()
        child.set_field_aggregator(self.field_aggregator)
        child.set_orientator(self.orientator)

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

        # Plagiotropism tolerance
        tol_rad = math.radians(self.options.plagiotropism_tolerance_angle)
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
