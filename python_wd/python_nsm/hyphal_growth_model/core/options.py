# core/options.py

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ToggleableFloat:
    enabled: bool = False
    value: float = 0.0

@dataclass
class ToggleableInt:
    enabled: bool = False
    value: int = 0

@dataclass
class Options:
    
    # ─── Core Simulation ───
    
    growth_rate: float = 1.0
    
    time_step: float = 1.0
    
    default_growth_vector: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=1.0))
    
    d_age: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=1.0))

    # ─── Branching Behavior ───
    
    branching_master: bool = True
    
    branch_probability: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.4))
    
    max_branches: ToggleableInt = field(default_factory=lambda: ToggleableInt(enabled=True, value=8))
    
    branch_angle_spread: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=180.0))
    
    leading_branch_prob: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=False, value=0.0))
    
    branch_sensitivity: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=1.0))
    
    branch_time_window: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=40.0))
    
    old_nbranch: bool = False
    
    secondary_branching: bool = True
    
    optimal_branch_orientation: bool = True

    # ─── Branch Crowding ───
    
    density_dependend: bool = False
    
    branching_density: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.06))
    
    complete_evaluation: bool = True
    
    log_branch_points: bool = False
    
    field_threshold: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=False, value=0.06))

    # ─── Tropisms ───
    
    autotropism: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=1.0))
    
    autotropism_impact: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=-1.0))
    
    field_hypothesis: bool = True
    
    gravitropism: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=False, value=1.0))
    
    random_walk: float = 0.2

    # ─── Growth Scaling & Curvature ───
    
    length_scaled_growth: bool = True
    
    length_growth_coef: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.1))
    
    curvature_branch_bias: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.25))

    # ─── Directional Memory ───
    
    direction_memory_blend: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.2))
    
    field_alignment_boost: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.2))
    
    field_curvature_influence: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.2))

    # ─── Age & Length Limits ───
    
    max_length: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=50.0))
    
    max_age: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=300.0))
    
    min_tip_age: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=10.0))
    
    min_tip_length: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=10.0))
    
    die_if_old: bool = False
    
    die_if_too_dense: bool = True
    
    min_supported_tips: ToggleableInt = field(default_factory=lambda: ToggleableInt(enabled=False, value=16))
    
    max_supported_tips: ToggleableInt = field(default_factory=lambda: ToggleableInt(enabled=False, value=1000000))
    
    plagiotropism_tolerance: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=5.0))

    # ─── Density Field ───
    
    density_field_enabled: bool = True
    
    density_threshold: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.2))
    
    charge_unit_length: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=20.0))
    
    neighbour_radius: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=400.0))
    
    density_from_tips: bool = True
    
    density_from_branches: bool = True
    
    density_from_all: bool = True

    # ─── Gravitropic Curvature ───
    
    gravi_angle_start: float = 100.0
    
    gravi_angle_end: float = 500.0
    
    gravi_angle_max: float = 180.0
    
    gravi_layer_thickness: float = 40.0

    # ─── Nutrient Fields ───
    
    use_nutrient_field: bool = False
    
    nutrient_attraction: float = 0.0
    
    nutrient_repulsion: float = 0.0
    
    nutrient_attract_pos: str = "30,30,0"
    
    nutrient_repel_pos: str = "-20,-20,0"
    
    nutrient_radius: float = 50.0
    
    nutrient_decay: float = 0.05
    
    nutrient_attractors: List[Tuple[Tuple[float, float, float], float]] = field(default_factory=lambda: [((30, 30, 0), 1.0)])
    
    nutrient_repellents: List[Tuple[Tuple[float, float, float], float]] = field(default_factory=lambda: [((-20, -20, 0), -1.0)])

    # ─── Anisotropy ───
    
    anisotropy_enabled: bool = False
    
    anisotropy_vector: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    
    anisotropy_strength: float = 0.1

    # ─── Debugging & I/O ───
    
    record_dead_tips: bool = True
    
    source_config_path: str = ""
    
    # ─── Reproducibility ───
    
    seed: int = 123
