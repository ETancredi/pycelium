# main.py

import matplotlib
matplotlib.use("Agg") 
import sys
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from core.mycel import Mycel
from core.point import MPoint
from core.options import Options

from tropisms.orientator import Orientator
from tropisms.nutrient_field_finder import NutrientFieldFinder

from compute.field_aggregator import FieldAggregator

from io_utils.checkpoint import CheckpointSaver
from io_utils.autostop import AutoStop
from io_utils.grid_export import export_grid_to_csv, export_grid_to_png
from io_utils.exporter import export_to_csv, export_to_obj, export_tip_history

from control.runtime_mutator import RuntimeMutator

from vis.density_map import DensityGrid, plot_density
from vis.plot2d import plot_mycel
from vis.plot3d import plot_mycel_3d
from vis.analyser import SimulationStats, plot_stats
from vis.nutrient_vis import plot_nutrient_field_2d, plot_nutrient_field_3d
from vis.anisotropy_grid import AnisotropyGrid, plot_anisotropy_2d, plot_anisotropy_3d
from vis.animate_growth import animate_growth
from vis.plotly_3d_export import plot_mycel_3d_interactive

from analysis.stats_summary import summarise
from analysis.post_analysis import analyse_branching_angles, analyse_tip_orientations

from config.sim_config import load_options_from_json

def setup_simulation(opts):
    # Set random seed for reproducibility
    if hasattr(opts, "seed"):
        print(f"🌱 Setting random seed: {opts.seed}")
        random.seed(opts.seed)
        np.random.seed(opts.seed)
    else:
        print("🌱 No seed specified; using system randomness.")

    # Create mycel instance
    mycel = Mycel(opts)

    def random_point_on_sphere(radius):
        theta = random.uniform(0,2 * math.pi)
        phi = math.acos(random.uniform(-1,1))
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        return MPoint(round(x), round(y), round(z))

    seed1 = MPoint(0,0,0)
    seed2 = random_point_on_sphere(radius=1)
    mycel.seed(seed1, seed2)

    orientator = Orientator(opts)
    aggregator = FieldAggregator()
    aggregator.set_options(opts)

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

    grid = DensityGrid(width=100, height=100, resolution=1.0)
    orientator.set_density_grid(grid)

    anisotropy_grid = None
    if opts.anisotropy_enabled:
        anisotropy_grid = AnisotropyGrid(width=100, height=100, depth=100, resolution=10.0)
        anisotropy_grid.set_uniform_direction(MPoint(*opts.anisotropy_vector))
        orientator.set_anisotropy_grid(anisotropy_grid)

    output_dir = os.getenv("BATCH_OUTPUT_DIR", "outputs")
    print(f"🔍 BATCH_OUTPUT_DIR is: {output_dir}")
    checkpoints_folder = os.path.join(output_dir, "checkpoints")
    checkpoints = CheckpointSaver(interval_steps=20, output_dir=checkpoints_folder)
    autostop = AutoStop(enabled=True)
    mutator = RuntimeMutator()
    stats = SimulationStats()

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
    aggregator = components["aggregator"]
    grid = components["grid"]
    orientator = components["orientator"]
    checkpoints = components["checkpoints"]
    autostop = components["autostop"]
    mutator = components["mutator"]
    stats = components["stats"]
    opts = components["opts"]

    print(f"\n🔄 STEP {step}")
    aggregator.sources.clear()
    aggregator.add_sections(mycel.get_all_segments(), strength=1.0, decay=1.5)

    for tip in mycel.get_tips():
        tip.orientation = orientator.compute(tip)

    mycel.step()
    grid.update_from_mycel(mycel)

    if opts.use_nutrient_field and opts.nutrient_repulsion > 0:
        mycel.nutrient_kill_check()

    mutator.apply(step, opts)
    checkpoints.maybe_save(mycel, step)
    stats.update(mycel)
    
    print(mycel)

def generate_outputs(mycel, components, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    grid = components["grid"]
    stats = components["stats"]
    opts = components["opts"]
    anisotropy_grid = components.get("anisotropy_grid", None)

    print(f"📸 Saving all plots to '{output_dir}'...")
    plot_mycel(mycel, title="2D Projection", save_path=f"{output_dir}/mycelium_2d.png")
    plot_mycel_3d(mycel, title="3D Projection", save_path=f"{output_dir}/mycelium_3d.png")
    plot_density(grid, save_path=f"{output_dir}/density_map.png")
    plot_stats(stats, save_path=f"{output_dir}/stats.png")
    plot_mycel_3d_interactive(mycel, save_path=f"{output_dir}/mycelium_3d_interactive.html")

    if opts.use_nutrient_field:
        plot_nutrient_field_2d(opts, save_path=f"{output_dir}/nutrient_2d.png")
        plot_nutrient_field_3d(opts, save_path=f"{output_dir}/nutrient_3d.png")

    if opts.anisotropy_enabled and anisotropy_grid:
        plot_anisotropy_2d(anisotropy_grid, save_path=f"{output_dir}/anisotropy_2d.png")
        plot_anisotropy_3d(anisotropy_grid, save_path=f"{output_dir}/anisotropy_3d.png")

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

    export_grid_to_csv(grid, f"{output_dir}/density_map.csv")
    export_to_csv(mycel, f"{output_dir}/mycelium_time_step.csv", all_time=True)
    export_to_csv(mycel, f"{output_dir}/mycelium_final.csv", all_time=False)
    export_to_obj(mycel, f"{output_dir}/mycelium.obj")
    export_tip_history(mycel, f"{output_dir}/mycelium_time_series.csv")
    
    animate_growth(
      csv_path=f"{output_dir}/mycelium_time_series.csv",
      save_path=f"{output_dir}/mycelium_growth.mp4",
      interval = 100 # ms per frame
    )

def simulate(opts, steps=120):
    mycel, components = setup_simulation(opts)

    try:
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
        print("\n🛑 Interrupted by user. Saving final state...")

    # Smart output folder detection
    if any("batch_runner.py" in arg for arg in sys.argv):
        # If running from batch_runner.py → place outputs in the batch_outputs folder
        output_dir = os.getenv("BATCH_OUTPUT_DIR", "outputs")
    else:
        output_dir = "outputs"

    print(f"📂 Saving outputs to: {output_dir}")
    generate_outputs(mycel, components, output_dir=output_dir)

if __name__ == "__main__":
    opts = load_options_from_json("configs/example.json")
    simulate(opts, steps=120)
