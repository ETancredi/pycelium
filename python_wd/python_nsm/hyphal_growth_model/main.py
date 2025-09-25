# main.py

# Imports
import matplotlib 
matplotlib.use("Agg") # Use non-interactive backend so figures can be saved without a display
import sys # Access to cmd-line args for mode detection
import os # Filesystem ops
import random # Python RNG for reproducible seeds
import math # Math utilities
import numpy as np # NumPy for numerical ops and seeding
import matplotlib.pyplot as plt # Plotting library

from core.mycel import Mycel # Main sim engine
from core.point import MPoint # 3D point / vector ops
from core.options import Options # Dataclass for all config opts

# Tropisms and nutrient field logic
from tropisms.orientator import Orientator
from tropisms.nutrient_field_finder import NutrientFieldFinder

# Field aggregation across various sources
from compute.field_aggregator import FieldAggregator

# I/O utils: checkpointing, auto-stop, grid-exports, data exporters
from io_utils.checkpoint import CheckpointSaver
from io_utils.autostop import AutoStop
from io_utils.grid_export import export_grid_to_csv, export_grid_to_png
from io_utils.exporter import export_to_csv, export_to_obj, export_tip_history, export_biomass_history
from io_utils.logging_utils import setup_logging, parse_int_env
logger = setup_logging("pycelium")  # default WARNING; override with PYCELIUM_LOG_LEVEL

# Runtime control and mutation of params
from control.runtime_mutator import RuntimeMutator

# Visualisation utilities
from vis.density_map import DensityGrid, plot_density
from vis.plot2d import plot_mycel
from vis.plot3d import plot_mycel_3d
from vis.analyser import SimulationStats, plot_stats
from vis.nutrient_vis import plot_nutrient_field_2d, plot_nutrient_field_3d
from vis.anisotropy_grid import AnisotropyGrid, plot_anisotropy_2d, plot_anisotropy_3d
from vis.animate_growth import animate_growth
from vis.plotly_3d_export import plot_mycel_3d_interactive

# Post-sim analysis
from analysis.stats_summary import summarise
from analysis.post_analysis import analyse_branching_angles, analyse_tip_orientations

# Config loader for CLI-mode
from config.sim_config import load_options_from_json

def setup_simulation(opts):
    """
    Initialise simulation: 
        set seeds, 
        create Mycel instance, 
        configure tropisms,
        grids,
        checkpoints.
        + other components.
    Returns:
        Mycel, components_dict
    """
    # Set random seed for reproducibility
    if hasattr(opts, "seed") and opts.seed is not None:
        logger.info(f"Seed: {opts.seed}")
        random.seed(opts.seed); np.random.seed(opts.seed)
    else:
        logger.info("Seed: <random or external>")

    # Instantiate main simulation engine
    mycel = Mycel(opts)

    # Helper to pick a random point on a sphere
    def random_point_on_sphere(radius): 
        theta = random.uniform(0,2 * math.pi) # aximuthal angle
        phi = math.acos(random.uniform(-1,1)) # polar angle
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        return MPoint(round(x), round(y), round(z))

    # Seed initial two tips: one at origin, one at random sphere point
    seed1 = MPoint(0,0,0) 
    seed2 = random_point_on_sphere(radius=1)
    mycel.seed(seed1, seed2, color=opts.initial_color)

    # Create orientator and field aggregator for tropism calculations
    orientator = Orientator(opts) 
    aggregator = FieldAggregator()
    aggregator.set_options(opts)

    # If nutrient field is used, add attractors/repellents to aggregator
    if opts.nutrient_attraction > 0: 
        aggregator.add_finder(NutrientFieldFinder(
            MPoint(30, 30, 0),
            strength=opts.nutrient_attraction,
            decay=1.0,
            repulsive=False
        ))
    if opts.nutrient_repulsion > 0:
        aggregator.add_finder(NutrientFieldFinder(
            MPoint(-30, -30, 0),
            strength=opts.nutrient_repulsion,
            decay=1.0,
            repulsive=True
        ))

    orientator.set_field_source(aggregator)

    # Initialise density grid for avoidance behaviours
    grid = DensityGrid(width=100, height=100, resolution=1.0) 
    orientator.set_density_grid(grid)

    # Optionally set up anisotropy grid if enabled
    anisotropy_grid = None 
    if opts.anisotropy_enabled:
        anisotropy_grid = AnisotropyGrid(width=100, height=100, depth=100, resolution=10.0)
        anisotropy_grid.set_uniform_direction(MPoint(*opts.anisotropy_vector))
        orientator.set_anisotropy_grid(anisotropy_grid)

    # Determine output directory from environment (batch or default)
    output_dir = os.getenv("BATCH_OUTPUT_DIR", "outputs") 
    logger.info(f"Output dir: {output_dir}")
    
    # Set up checkpoint saver to write JSON every N steps
    checkpoints_folder = os.path.join(output_dir, "checkpoints")
    checkpoints = CheckpointSaver(interval_steps=20, output_dir=checkpoints_folder)
    
    # Autostop monitor, runtime mutator, and stats collector
    autostop = AutoStop(enabled=True)
    mutator = RuntimeMutator()
    stats = SimulationStats()

    # Bundle all components into a dict for stepping
    return mycel, {
        "orientator": orientator,
        "aggregator": aggregator,
        "grid": grid,
        "checkpoints": checkpoints,
        "autostop": autostop,
        "mutator": mutator,
        "stats": stats,
        "opts": opts,
        "anisotropy_grid": anisotropy_grid
    }

def step_simulation(mycel, components, step):
    """
    Perform one timestep: 
        Update tropism fields,
        Apply orientator,
        Step the Mycel model,
        Update density grid,
        Apply mutator (if enabled),
        Checkpoints, stats
    """
    # Unpack components for convenience
    aggregator = components["aggregator"]
    grid = components["grid"]
    orientator = components["orientator"]
    checkpoints = components["checkpoints"]
    autostop = components["autostop"]
    mutator = components["mutator"]
    stats = components["stats"]
    opts = components["opts"]

    print(f"\n🔄 STEP {step}")
    # C;ear previous field sources and re-add all sections as SectFieldFinders
    aggregator.sources.clear()
    aggregator.add_sections(mycel.get_all_segments(), strength=1.0, decay=1.5)

    # Compute new orientation for each tip using orientator
    for tip in mycel.get_tips():
        tip.orientation = orientator.compute(tip)

    # Advance simulation by one time step (grow, branch, prune)
    mycel.step()
    # Update density grid counts from all segment ends
    grid.update_from_mycel(mycel)

    # Optionally perform nutrient-based kill check if implemented
    if opts.use_nutrient_field and opts.nutrient_repulsion > 0:
        mycel.nutrient_kill_check()

    # Apply any schedules parameter mutations at this step
    mutator.apply(step, opts)
    # Save a acheckpoint if interval reached
    checkpoints.maybe_save(mycel, step)
    # Record stats for plotting later
    stats.update(mycel)
    
    # Print a summary string of the current simulation state
    print(mycel)

def generate_outputs(mycel, components, output_dir="outputs"):
    """
    After simulation ends, generate all plots, exports, and analyses into the specific output dir
    """
    os.makedirs(output_dir, exist_ok=True)

    # Unpack frequently used components
    grid = components["grid"]
    stats = components["stats"]
    opts = components["opts"]
    anisotropy_grid = components.get("anisotropy_grid", None)

    print(f"📸 Saving all plots to '{output_dir}'...")

    # 2D and 3D static plots
    plot_mycel(mycel, title="2D Projection", save_path=f"{output_dir}/mycelium_2d.png")
    plot_mycel_3d(mycel, title="3D Projection", save_path=f"{output_dir}/mycelium_3d.png")
    plot_density(grid, save_path=f"{output_dir}/density_map.png")
    plot_stats(stats, save_path=f"{output_dir}/stats.png")
    plot_mycel_3d_interactive(mycel, save_path=f"{output_dir}/mycelium_3d_interactive.html")

    # Nutrient field visualisations if enabled
    if opts.use_nutrient_field:
        plot_nutrient_field_2d(opts, save_path=f"{output_dir}/nutrient_2d.png")
        plot_nutrient_field_3d(opts, save_path=f"{output_dir}/nutrient_3d.png")

    # Anisotropy grid visualisations if enabled
    if opts.anisotropy_enabled and anisotropy_grid:
        plot_anisotropy_2d(anisotropy_grid, save_path=f"{output_dir}/anisotropy_2d.png")
        plot_anisotropy_3d(anisotropy_grid, save_path=f"{output_dir}/anisotropy_3d.png")

    # Post-analysis: branching andles and tip orientations
    analyse_branching_angles(
        mycel,
        save_path=f"{output_dir}/branching_angles.png",
        csv_path=f"{output_dir}/branching_angles.csv"
    )
    analyse_tip_orientations(
        mycel,
        save_path=f"{output_dir}/tip_orientations.png",
        csv_path=f"{output_dir}/orientations.csv"
    )

    # Export data tables and 3D geometry
    export_grid_to_csv(grid, f"{output_dir}/density_map.csv")
    export_to_csv(mycel, f"{output_dir}/mycelium_time_step.csv", all_time=True)
    export_to_csv(mycel, f"{output_dir}/mycelium_final.csv", all_time=False)
    export_to_obj(mycel, f"{output_dir}/mycelium.obj")
    export_tip_history(mycel, f"{output_dir}/mycelium_time_series.csv")
    
    # Create a video or GIF of tip positions over time
    animate_growth(
      csv_path=f"{output_dir}/mycelium_time_series.csv",
      save_path=f"{output_dir}/mycelium_growth.mp4",
      interval = 100 # ms per frame
    )

    # Export biomass and tip history
    export_tip_history(mycel, f"{output_dir}/mycelium_time_series.csv")
    export_biomass_history(mycel, f"{output_dir}/biomass_and_tips_history.csv")

def simulate(opts, steps=120):
    """
    Top-level function to run a full sim loop, handle autostop, and then call generate_outputs at the end.
    """
    # Initialise sim and components
    mycel, components = setup_simulation(opts)

    try:
        # Loop for the requested no. steps
        for step in range(steps):
            step_simulation(mycel, components, step)

            # DEBUG: Print current tip states before AutoStop
            print(f"🛠️ DEBUG: Before AutoStop at step {step}")
            print(f"Mycel sections: {len(mycel.sections)}")
            alive_tips = [s for s in mycel.sections if s.is_tip and not s.is_dead]
            for idx, s in enumerate(mycel.sections):
                print(f"  Section {idx}: is_tip={s.is_tip}, is_dead={s.is_dead}, children={len(s.children)}")

            print(f"🛠️ DEBUG: Alive tips count = {len(alive_tips)}")

            # Check AutoStop condition
            if components["autostop"].check(mycel, step):
                print("🛠️ DEBUG: AutoStop triggered the stop.")
                break

    except KeyboardInterrupt:
        # Allow user to interrupt simulation with Ctrl+C and still save results
        print("\n🛑 Interrupted by user. Saving final state...")

    # Determine final output folder:
    # Always prefer BATCH_OUTPUT_DIR if set (batch, CLI, or GUI),
    # otherwise fall back to plain "outputs".
    output_dir = os.getenv("BATCH_OUTPUT_DIR", "outputs")
    
    print(f"📂 Saving outputs to: {output_dir}")
    generate_outputs(mycel, components, output_dir=output_dir) # Generate all plots and exports

if __name__ == "__main__":
    # If run directly, load a default config and simulate
    opts = load_options_from_json("configs/example.json")
    simulate(opts, steps=120)
