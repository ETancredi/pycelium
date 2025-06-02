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
    
    # The base linear growth speed (units per time-step)
    # Passed into Section.grow(rate, dt)
    # Controls how fast each tip extends, before any scaling
    
    time_step: float = 1.0

    # The "dt" used for each step
    # Sections grow by (growth_rate * time_step)
    # Also used for ageing ("age += dt") and timestamping
    
    default_growth_vector: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=1.0))

    # A global scalar multiplier on every step's growth distance
    # If enabled == True, each grow distance is multiplied by this value
    
    d_age: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=1.0))

    # Age-based slowdown exponent
    # In Section.grow: 
    #     rate /= (age ** d_age.value) | rate is equal to rate divided by (age to the power of the value of d_age)
    # Meaning that older tips slow down as age increases (when d_age is toggled on)
    # If enabled == False, this age-slowdown is skipped (i.e. age^0 = 1) so rate = rate / (age ** 0) = rate / 1

    # ─── Branching Behavior ───
    
    branching_master: bool = True

    # A global ON/OFF switch for _all_ branching
    # If bool = False, Section.maybe_branch() immediately return, no tip ever branches
    
    branch_probability: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.4))

    # The per-step "coin-flip" probability that a tip will attempt a branch
    # In Section.maybe_branch(), you draw r = rand(); if r < branch_probability.value, you proceed
    # If enabled == False, that gate is effectively "0.0" (i.e. no branching occurs becuase r > 0)
    
    max_branches: ToggleableInt = field(default_factory=lambda: ToggleableInt(enabled=True, value=8))

    # The maximum no. children a given Section is allowed to create
    # Each Section keeps track of `branches_made`
    # Once that reaches max_branches.value (if enabled == True), no more branches will be created from that section
    
    branch_angle_spread: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=180.0))

    # The maximum angle (in degrees) from the parent's direction when picking a new branch angle
    # If enabled == False, you treat that value as 0 (essentially no branching).
    
    leading_branch_prob: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=False, value=0.0))

    # When a branch is created, you randomly choose whether the child becomes the new leading tip or the parent continues straight and the child is diverted
    # If rand() < leading_branch_prob.value, child keeps rotated orientation
    # If rand() > leading_branch_prob.value, parent rotates and child keeps old orientation
    # If enabled == False (or value == 0), parent always "yields" its orientation to new branch
    
    branch_sensitivity: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=1.0))

    # Not currently integrated into model
    # Supposed to describe scaling how strongly tips respond to a field when branching
    
    old_nbranch: bool = False

    # If True, any tip older than branch_time_window.value is not allowd to branch, although it still lives until max_age
    # If False, 'branch_time_window' is ignorred
    
    branch_time_window: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=40.0))

    # If `old_nbranch` is enabled, this is the cutoff age
    # In Section.maybe_branch():
    #    if old_nbranch = True and age > branch_time_window.value: return None
    #        i.e. after this age, tips can no longer branch (but will remain alive)
    
    secondary_branching: bool = True

    # If False, a node that has already spawns one child will not spawn any more
    # In Section.maybe_branch():
    #    if self.children and (not secondary_branching): return None
    #         i.e. this flag allows/disallows "branching off of a branch" (beyond the first fork)
    
    optimal_branch_orientation: bool = True

    # If True, in Section.get_new_growing_vector() the code first asks the compute.field_aggregator:
    #     strength, field_vec = compute_field(...)
    #         and if field_vec is not 0, it uses that direction instead of random-cone sampling.

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
