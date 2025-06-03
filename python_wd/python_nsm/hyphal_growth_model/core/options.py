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
        # If False, branches always come from uniform cone-sampling

    # ─── Branch Crowding ───
    density_dependend: bool = False
        # If True, every tip checks local hyphal density before branching
        #     If compute_local_hyphal_density(radius) ≥ brasnching_density.value --> no branch.
        # If False, skip hyphal-crowding gate entirely
    
    branching_density: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.06))
        # The threshold, in "field units", above which tips refuse to branch if density_dependend==True
        # Local density is computed by summing the decaying contributions of all other SectFieldFinder sources within neighbour_radius
        
    complete_evaluation: bool = True
        # Controls how many subsegment-locations you sample when checking the environmental field in maybe_branch():
        #     if True, samples every subsegment's endpoint
        #     if False, samples only the very tip ends
        # (more sample points -> more conservative gating)
        
    log_branch_points: bool = False
        # If True, you also add the first branch-junction location to the list of points used in environmental-field gating
        # More of a debugging tool
    
    field_threshold: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=False, value=0.06))
        # Scalar cutoff on "environmental field strength"
        # In maybe_branch:
        #     strengths = [ compute_field(pt)[0] for pt in points ]
        #     if max(strengths) ≥ field_threshold.value: return None
        # i.e. If local external field is too strong, block branching

    # ─── Tropisms ───
    autotropism: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=1.0))
        # In orientator.compute(), you do two things with autotropism:  
        #   1) If field_hypothesis==True, add grad*strength.  
        #   2) Independently (always) add (orientation * autotropism.value * autotropism_impact.value).  
        # This term tends to keep the tip aligned along its current direction (self‐avoidance).    
        
    autotropism_impact: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=-1.0))
        # Multiplies the “autotropism” force. Typically negative so that tips are repelled from existing sections.  
        # In orientator.compute():  
        #    orientation.add( section.orientation.scale(autotropism.value * autotropism_impact.value) )
    
    field_hypothesis: bool = True
        # When True, orientator first queries the field aggregator at the tip, gets (strength, grad), and does  
        #     orientation.add(grad.scale(strength))  
        # In effect, tips move up or down field gradients.
    
    gravitropism: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=False, value=1.0))
        # If enabled, tips feel a constant “pull” toward –Z (downwards).  
        # In orientator.compute(): you compute  
        #     if z<gravi_angle_start → strength=0  
        #     elif z>gravi_angle_end → strength=gravitropism.value  
        #     else linearly ramp.  
        # then add (0,0,–1)*strength to the orientation.
    
    random_walk: float = 0.2
        # A plain (non‐toggleable) jitter‐magnitude. At each orient step, you sample a unit random vector 
        # and add (unit_rand * random_walk) to the orientation.

    # ─── Growth Scaling & Curvature ───
    
    length_scaled_growth: bool = True
        # If True, in Section.grow you multiply rate by (1 + length * length_growth_coef).  
        # This makes longer hyphae grow even faster (or slower if coef<0).
    
    length_growth_coef: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.1))
        # The “length” coefficient in length‐scaled growth.  
        # If enabled, scale_factor = 1 + length * length_growth_coef.value.
        
    curvature_branch_bias: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.25))
        # In maybe_branch(), after selecting a rotated_orientation, you blend it with the local “curvature vector” computed from the last 3 subsegments:  
        #     new_dir = (rotated * (1–bias) + curve*(bias)).normalize()  
        # If enabled, this introduces a bias that keeps new branches following the existing curve.
    
    # ─── Directional Memory ───
    
    direction_memory_blend: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.2))
        # In Section.grow (after updating orientation), you update  
        #     direction_memory = (direction_memory*(1–α) + orientation*α).normalize()  
        # In orientator.compute, at the end you re‐blend:  
        #     orientation = (old_orientation*α + orientation*(1–α)).normalize()  
        # so that tips “remember” their recent direction.
        
    field_alignment_boost: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.2))
        # After adding the field gradient to orientation, you compute “dot = orientation·grad_unit.”  
        # If dot>0, you add grad_unit*(dot * field_alignment_boost.value).  
        # This gives an extra nudge to align tightly with the field in the direction already moving.
    
    field_curvature_influence: ToggleableFloat = field(default_factory=lambda: ToggleableFloat(enabled=True, value=0.2))
        # In orientator.compute, you call compute_field_curvature(section.end) (a Laplacian approximation).  
        # You multiply that curvature by field_curvature_influence.value and add (grad_unit * that).  
        # This injects second‐derivative cues from the field (peaks/valleys) into the heading.
    
    # ─── Age & Length Limits ───
    max_length: ToggleableFloat = ToggleableFloat(enabled=True, value=50.0)
        # If enabled, any tip whose length exceeds max_length.value is killed in the destructor pass.  
        # If disabled, the code treats max_length as “no length limit.”

    max_age: ToggleableFloat = ToggleableFloat(enabled=True, value=300.0)
        # If enabled, any tip whose age exceeds max_age.value is killed outright in the destructor pass.  
        # If disabled, tips do not die from age alone.

    min_tip_age: ToggleableFloat = ToggleableFloat(enabled=True, value=10.0)
        # (Currently not used, because “min_tip_age” was replaced by the old “enforce_min_tip_age”/“enforce_min_tip_length.”)  
        # In the Java NSM this would block branching if age < min_tip_age.value.  (You can remove or re‐implement if needed.)

    min_tip_length: ToggleableFloat = ToggleableFloat(enabled=True, value=10.0)
        # (Also not used in the Python port as written.)  Would block branching until a tip’s length ≥ this.  
        # You can delete if you never reference it.

    die_if_old: bool = False
        # If True, in the destructor loop you kill any tip older than max_age (or max_age.value).  
        # If False, you skip “death by age” entirely.

    die_if_too_dense: bool = True
        # If True, in the destructor you compute `density = field_aggregator.compute_field(end)[0]`.  
        # If density > density_threshold, you kill the tip.  If False, skip that density‐kill check.

    min_supported_tips: ToggleableInt = ToggleableInt(enabled=True, value=16)
        # In the destructor’s “neighborhood support” check, you count how many *other* tips lie within neighbour_radius of this tip.  
        # If enabled and nearby_count < min_supported_tips.value, you kill this tip for being isolated.

    max_supported_tips: ToggleableInt = ToggleableInt(enabled=True, value=1000000)
        # Similarly, if enabled and nearby_count > max_supported_tips.value, you kill the tip for overcrowding.  
        # (Often you leave this off or set it very large so it never triggers.)

    plagiotropism_tolerance_angle: ToggleableFloat = ToggleableFloat(enabled=True, value=5.0)
        # In get_new_growing_vector(), after computing a candidate new_dir, you check its angle relative to pure “down” (–Z).  
        # If the angle exceeds tolerance, you clamp new_dir to straight downward.  
        # This enforces a “plagiotropic” rule: branches cannot exceed X° from vertical.

    # ─── Density Field ───
    density_field_enabled: bool = True
        # If False, the orientator never uses the density grid’s gradient for self‐avoidance.  
        # (I.e. skip the “if die_if_too_dense and density_grid...” step in orientator.)

    density_threshold: ToggleableFloat = ToggleableFloat(enabled=True, value=0.2)
        # The cutoff used in the destructor’s “density kill” check.  
        # Only applies if die_if_too_dense==True.  
        # If density > density_threshold.value, kill the tip.

    charge_unit_length: ToggleableFloat = ToggleableFloat(enabled=True, value=20.0)
        # When building a density grid (DensityGrid), this is the length (in world units) that corresponds to one unit charge.  
        # Used in computing per‐tip contribution to the density map. (If you don’t use the Python DensityGrid, you can ignore this.)

    neighbour_radius: ToggleableFloat = ToggleableFloat(enabled=True, value=400.0)
        # The radius (in world units) used in:  
        #   • maybe_branch’s hyphal‐crowding check (compute_local_hyphal_density)  
        #   • destructor’s “neighborhood support” check (count tips within this radius)  
        # If disabled, treat as 0 (no neighbors counted → every tip is isolated).

    density_from_tips: bool = True
        # If building a density grid that tracks individual tips, include each tip as a density source.

    density_from_branches: bool = True
        # If building a density grid that also tracks whole branch segments as sources, include them.

    density_from_all: bool = True
        # If you want to combine both tips and branches into the single density field.  
        # (Controls whether you call add_sections() in FieldAggregator.)

    # ─── Gravitropic Curvature ───
    gravi_angle_start: float = 100.0
        # Height (Z) below which there is zero gravitropism pull.

    gravi_angle_end: float = 500.0
        # Height (Z) above which gravitational pull is at full `gravitropism.value`.  
        # Between start/end you ramp linearly.

    gravi_angle_max: float = 180.0
        # (Not actually used in your Python code.)  Could cap the maximum turning angle due to gravity.

    gravi_layer_thickness: float = 40.0
        # (Also not used in this port.)  In Java NSM it defined a “layer” height for curvature, e.g. where gravitropism blends.

    # ─── Nutrient Fields ───
    use_nutrient_field: bool = False
        # If True, after Section.step you’ll do `if use_nutrient_field and field<–abs(nutrient_repulsion) → kill tip.`  
        # Also, earlier in setup_simulation() you added NutrientFieldFinder sources to the aggregator.

    nutrient_attraction: float = 0.0
        # If >0, you add a NutrientFieldFinder source at (nutrient_attract_pos) with this strength and decay=1.0  
        # Tips will then “feel” attraction toward that point (in orientator).

    nutrient_repulsion: float = 0.0
        # If >0, you add a NutrientFieldFinder source at (nutrient_repel_pos) with negative strength.  
        # Tips are repelled from that location.

    nutrient_attract_pos: str = "30,30,0"
        # GUI‐friendly string to specify the (x,y,z) position of the attractor. Parsed by sim_config or GUI.

    nutrient_repel_pos: str = "-20,-20,0"
        # GUI string for the repellent’s (x,y,z).

    nutrient_radius: float = 50.0
        # In orientator.compute(), when processing nutrient sources, you only influence orientation if dist < nutrient_radius.

    nutrient_decay: float = 0.05
        # (Not currently used in your orientator logic; you hard‐coded decay=1.0 for NutrientFieldFinder.)

    nutrient_attractors: List[Tuple[Tuple[float,float,float], float]] =
        field(default_factory=lambda: [((30,30,0), 1.0)])
        # A Python list of (pos3D, strength) tuples for “legacy” attractors.  
        # GUI’s “Nutrient Editor” allows you to add/remove from this list.

    nutrient_repellents: List[Tuple[Tuple[float,float,float], float]] =
        field(default_factory=lambda: [((-20,-20,0), -1.0)])
        # Similar list for repellent points.

    # ─── Anisotropy ───
    anisotropy_enabled: bool = False
        # If True, orientator either queries an AnisotropyGrid (per‐voxel direction) or 
        # simply adds the global anisotropy_vector.  
        # If False, skip any anisotropy.

    anisotropy_vector: Tuple[float,float,float] = (1.0,0.0,0.0)
        # If no grid is provided, orientator falls back to this uniform direction.

    anisotropy_strength: float = 0.1
        # How strongly to add anisotropy_vector to the tip’s orientation each step.

    # ─── Debugging & I/O ───
    record_dead_tips: bool = True
        # If True, the code keeps track of dead tips in a list/log.  
        # (Right now you don’t explicitly use it in core code, but could toggle extra logging.)

    source_config_path: str = ""
        # If you ever load an older NSM “.xml” or custom JSON, you can point to its path here.

    # ─── Reproducibility ───
    seed: int = 123
        # Random‐seed used to initialize `random.seed(seed)` and `np.random.seed(seed)` at startup.
