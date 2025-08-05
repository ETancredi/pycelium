# core/options.py

# Imports
from dataclasses import dataclass, field # Dataclass decorator and field factory function for default mutable fields
from typing import List, Tuple # Import list and Tuple for tupe annotations of parameters

@dataclass
class Options:
    """Configuration container for all simulation parameters."""
    
    # Core simulation timing
    growth_rate: float = 1.0 # How quickly each hyphal segment increases per unit time
    time_step: float = 1.0 # Size of each discrete simulation step (Δt)

    # Branching behaviour
    branch_probability: float = 0.4 # Probability that a tip will attempt to branch in a single time step                 
    max_branches: int = 8 # Max. no. simultaneous branches allowed per segment                          
    branch_angle_spread: float = 180.0 # Max. angular deviation (in degrees) from parent orientation for new branches              
    field_threshold: float = 0.06 # Minimum substrate / chemical field strength required to allow branching                  
    branch_time_window: float = 40.0 # Time window (age) during which branching is permitted for a segment
    branch_sensitivity: float = 1.0 # Scaling factor applied to field influences when deciding to branch
    optimise_initial_branching: bool = True # If True, uses a heuristic to favour early branching for root establishment
    leading_branch_prob: float = 0.0 # Extra probability boost assigned to the first branch of each tip
    allow_internal_branching: bool = True # If True, non-tip segments can also form new branches           

    # Tropisms and Growth Modifiers
    autotropism: float = 1.0 # Strength of self-alignment: tendency to continue in current direction
    gravitropism: float = 0.0 # Strength of gravity driven curvature (positive values bend downward)
    random_walk: float = 0.4 # Magnitude of stochastic perturbation added to growth direction
    length_scaled_growth: bool = True # If True, growth_rate is scaled by segment length
    length_growth_coef: float = 0.1 # Coefficient for length-based growth scaling
    curvature_branch_bias: float = 0.25  # Bias factor: move curved segments are more likely to branch
    
    # Directional memory (EMA decay)
    direction_memory_blend: float = 0.1 # Exponential moving avg. blend factor for past direction vs. new influences
    field_alignment_boost: float = 0.2 # Weight given to aligning with external chemical/substrate fields 
    field_curvature_influence: float = 0.2 # Degree to which local field curvature steers the growth direction

    # Age & Length limitations
    max_length: float = 50.0 # If segment length exceeds this, it dies            
    die_if_old: bool = False # If True, segments older than max_age die
    max_age: float = 300.0 # Max. age (in sim time units) before age-based death              
    min_tip_age: float = 10.0 # Min. age before a tip is allowed to branch
    min_tip_length: float = 10.0 # Min. length before a tip is allowed to branch
    die_if_too_dense: bool = True # If True, segments in overly dense regions )based on density field) are killed
    min_supported_tips: int = 16 # Min. no. neighbouring tips required for survival
    max_supported_tips: int = 1000 # Max. total active tips allowed before pruning occurs      

    # Density field
    density_field_enabled: bool = True # Toggle for computing a crowding field from all sources
    density_threshold: float = 0.2 # Field value above which sgements are considered too dense
    charge_unit_length: float = 20.0 # Scaling factor for how much "charge" each segment contributed to the field
    neighbour_radius: float = 400.0 # Radius (distance) w/in which other tips count towards density
    density_from_tips: bool = True # Include tip contributions in the density field
    density_from_branches: bool = True # Include branch (non-tip segment) contributions
    density_from_all: bool = True # Include all segments when computing density       

    # Gravitropism curvature (angle-based)
    gravi_angle_start: float = 100.0 # Min. angle (degrees) at which gravity begins to influence curvature
    gravi_angle_end: float = 500.0 # Angle above which maximum gravitropic curvature is applied
    gravi_angle_max: float = 180.0 # Max. curvature angle change allowed per step
    gravi_layer_thickness: float = 40.0 # Thickness of layers used to modulate gravitropic response

    # Data recording & debugging
    record_dead_tips: bool = True # If True, positions of dead tips are still recorded
    source_config_path: str = "" # Optional path ot YAML/JSON file for external parameter loading

    # Nutrient field settings
    use_nutrient_field: bool = False # Enable conputation of a separate nutrient concentration field
      
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

    # Volume Constraint
    volume_constraint: bool = False
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = -10.0
    z_max: float = 10.0

    # RGB Mutation Settings
    rgb_mutations_enabled: bool = False
    initial_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    color_mutation_prob: float = 0.05       # chance each daughter mutates
    color_mutation_scale: float = 0.02      # Laplace “b” parameter
    
    # Reproducibility
    seed: int = 123
