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

    # Color trait
    color: tuple  # (r, g, b)
    rgb_mutations_enabled: bool

    # Phenotype-level mutation rate (mutable)
    mutation_prob: float

    # Mutation ancestry flags
    mutated_from_seed: bool = False
    mutated_from_parent: bool = False

    def copy_with_mutation(self, mutation_scale: float, seed_phenotype=None):
        """
        Return a mutated copy of this phenotype.
        mutation_scale: Laplace scale parameter for mutations.
        seed_phenotype: used to track mutated_from_seed
        """
        new_pheno = copy.deepcopy(self)
        new_pheno.mutated_from_parent = False
        new_pheno.mutated_from_seed = False

        for field_name, value in asdict(self).items():
            # Skip flags and boolean mutation toggle
            if field_name in ("mutated_from_seed", "mutated_from_parent", "rgb_mutations_enabled"):
                continue

            if random.random() < self.mutation_prob:
                # Laplace-distributed mutation
                delta = np.random.laplace(0.0, mutation_scale)

                if isinstance(value, int):
                    new_value = max(0, round(value + delta))
                elif isinstance(value, float):
                    new_value = max(0.0, value + delta)
                elif isinstance(value, tuple):
                    # Mutate each channel of color
                    new_value = tuple(min(max(c + delta, 0.0), 1.0) for c in value)
                else:
                    # Skip any other types
                    continue

                setattr(new_pheno, field_name, new_value)
                new_pheno.mutated_from_parent = True

        # Track mutation relative to seed phenotype
        if seed_phenotype and new_pheno != seed_phenotype:
            new_pheno.mutated_from_seed = True

        return new_pheno

    def to_dict(self):
        return asdict(self)

    def __eq__(self, other):
        if not isinstance(other, Phenotype):
            return False
        # Compare all trait fields except flags
        return all(
            getattr(self, f) == getattr(other, f)
            for f in self.__dataclass_fields__
            if f not in ("mutated_from_seed", "mutated_from_parent")
        )
