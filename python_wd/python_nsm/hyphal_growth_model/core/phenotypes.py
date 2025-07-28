# core/phenotypes.py

import random
import copy
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class Phenotype:
    # Mutable traits
    growth_rate: float
    branch_probability: float
    max_branches: int
    branch_angle_spread: float
    field_threshold: float
    branch_time_window: float
    branch_sensitivity: float
    leading_branch_prob: float
    autotropism: float
    length_growth_coef: float
    curvature_branch_bias: float
    direction_memory_blend: float
    field_alignment_boost: float
    field_curvature_influence: float
    max_length: float
    max_age: float
    min_tip_age: float
    min_tip_length: float
    density_threshold: float
    charge_unit_length: float
    neighbour_radius: float

    # RGB mutation traits
    color: tuple  # (r, g, b)
    rgb_mutations_enabled: bool
    color_mutation_prob: float
    color_mutation_scale: float

    # Mutation ancestry flags
    mutated_from_seed: bool = False
    mutated_from_parent: bool = False

    def copy_with_mutation(self, mutation_prob=0.1, mutation_scale=0.05, seed_phenotype=None):
        """
        Return a mutated copy of this phenotype.
        mutation_prob: chance of mutating each trait
        mutation_scale: Laplace scale parameter (b)
        seed_phenotype: used to track mutated_from_seed
        """
        new_pheno = copy.deepcopy(self)
        new_pheno.mutated_from_parent = False
        new_pheno.mutated_from_seed = False

        for field_name, value in asdict(self).items():
            if field_name in ("color", "rgb_mutations_enabled", "color_mutation_prob", "color_mutation_scale",
                              "mutated_from_seed", "mutated_from_parent"):
                continue
            if random.random() < mutation_prob:
                mutation = np.random.laplace(0.0, mutation_scale)
                new_value = value + mutation
                # Clamp or round if needed
                if isinstance(value, int):
                    new_value = max(0, round(new_value))
                elif isinstance(value, float):
                    new_value = max(0.0, new_value)
                setattr(new_pheno, field_name, new_value)
                new_pheno.mutated_from_parent = True

        # RGB color mutation
        if self.rgb_mutations_enabled and random.random() < self.color_mutation_prob:
            r, g, b = self.color
            dr = np.random.laplace(0.0, self.color_mutation_scale)
            dg = np.random.laplace(0.0, self.color_mutation_scale)
            db = np.random.laplace(0.0, self.color_mutation_scale)
            new_pheno.color = (
                min(max(r + dr, 0.0), 1.0),
                min(max(g + dg, 0.0), 1.0),
                min(max(b + db, 0.0), 1.0)
            )
            new_pheno.mutated_from_parent = True

        if seed_phenotype and new_pheno != seed_phenotype:
            new_pheno.mutated_from_seed = True

        return new_pheno

    def to_dict(self):
        return asdict(self)

    def __eq__(self, other):
        if not isinstance(other, Phenotype):
            return False
        # Compare traits (not flags)
        return all(
            getattr(self, f) == getattr(other, f)
            for f in self.__dataclass_fields__
            if f not in ("mutated_from_seed", "mutated_from_parent")
        )
