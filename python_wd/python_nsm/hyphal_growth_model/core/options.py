# core/options.py

from dataclasses import dataclass, field

@dataclass
class Options:
    
    # Core simulation parameters
    growth_rate: float = 1.0
    time_step: float = 1.0
    default_growth_vector: float = 1.0
    d_age: float = 1.0

    # Branching behaviour
    branching_master: bool = True
    branch_probability: float = 0.4                 
    max_branches: int = 8                           
    branch_angle_spread: float = 180.0
    leading_branch_prob: float = 0.0
    branch_sensitivity: float = 1.0
    branch_time_window: float = 40.0
    secondary_branching: bool = False
    density_dependend: bool = True
    field_threshold: float = 0.06                   
    branching_density: float = 0.06
    old_nbranch: bool = False
    optimise_initial_branching: bool = True  
    optimal_branch_orientation: bool = False

    # Tropisms
    autotropism: float = 1.0
    gravitropism: float = 0.0
    random_walk: float = 0.2
    length_scaled_growth: bool = True
    length_growth_coef: float = 0.1
    curvature_branch_bias: float = 0.25 
    
    # Directional memory (EMA decay)
    direction_memory_blend: float = 0.2  
    field_alignment_boost: float = 0.2  
    field_curvature_influence: float = 0.2

    # Age & Length limitations
    max_length: float = 50.0            
    max_age: float = 300.0              
    min_tip_age: float = 10.0
    min_tip_length: float = 10.0
    die_if_old: bool = False
    die_if_too_dense: bool = True
    min_supported_tips: int = 16
    max_supported_tips: int = 1000      

    # Density field
    density_field_enabled: bool = True
    density_threshold: float = 0.2
    charge_unit_length: float = 20.0
    neighbour_radius: float = 400.0
    density_from_tips: bool = True
    density_from_branches: bool = True
    density_from_all: bool = True       

    # Gravitropism curvature (angle-based)
    gravi_angle_start: float = 100.0
    gravi_angle_end: float = 500.0
    gravi_angle_max: float = 180.0
    gravi_layer_thickness: float = 40.0

    # Data & debugging
    record_dead_tips: bool = True
    source_config_path: str = ""

    # Nutrient field settings
    use_nutrient_field: bool = False
      
    # Legacy-style attractors/repellents (for advanced use or CLI setup)
    nutrient_attractors: list = field(default_factory=lambda: [((30, 30, 0), 1.0)])  # (pos3D, strength)
    nutrient_repellents: list = field(default_factory=lambda: [((-20, -20, 0), -1.0)])
      
    # GUI-friendly default single field (with user inputs)
    nutrient_attraction: float = 0.0                 
    nutrient_repulsion: float = 0.0                  
    nutrient_attract_pos: str = "30,30,0"            
    nutrient_repel_pos: str = "-20,-20,0"            
    nutrient_radius: float = 50.0                    
    nutrient_decay: float = 0.05                     

    # Anisotropy
    anisotropy_enabled: bool = False
    anisotropy_vector: tuple = (1.0, 0.0, 0.0)  
    anisotropy_strength: float = 0.1            
    
    # Reproducibility
    seed: int = 123

    complete_evaluation: bool = True
    log_branch_points: bool = False
    field_hypothesis: bool = True
    autotropism_impact: float = -1.0
    plagiotropism_tolerance_angle: float = 5.0
