# main.py

# Imports
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend so figures can be saved without a display
import sys             # Access to cmd-line args for mode detection
import os              # Filesystem ops
import random          # Python RNG for reproducible seeds
import math            # Math utilities
import numpy as np     # NumPy for numerical ops and seeding
import matplotlib.pyplot as plt  # Plotting library

from core.mycel import Mycel              # Main sim engine
from core.point import MPoint             # 3D point / vector ops
from core.options import Options          # Dataclass for all config opts

# Tropisms and nutrient field logic
from tropisms.orientator import Orientator
from tropisms.nutrient_field_finder import NutrientFieldFinder

# Field aggregation across various sources
from compute.field_aggregator import FieldAggregator
from compute.drug_field import DrugField2D  # Diffusing antifungal field sampled by tips during growth

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
from vis.drug_overlay import plot_mycelium_on_drug_field
from vis.plot3d import plot_mycel_3d
from vis.analyser import SimulationStats, plot_stats
from vis.nutrient_vis import plot_nutrient_field_2d, plot_nutrient_field_3d
from vis.anisotropy_grid import AnisotropyGrid, plot_anisotropy_2d, plot_anisotropy_3d
from vis.animate_growth import animate_growth, animate_growth_2d, animate_network_growth_2d
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
        checkpoints,
        optional drug field,
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

    # Helpers to pick random seed positions
    def random_point_on_circle(radius: float) -> MPoint:
        """
        2D helper: pick a point uniformly on a circle of given radius in the z=0 plane.
        """
        theta = random.uniform(0, 2 * math.pi)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        return MPoint(round(x), round(y), 0.0)

    def random_point_on_sphere(radius: float) -> MPoint:
        """
        3D helper: pick a point uniformly on a sphere of given radius.
        """
        theta = random.uniform(0, 2 * math.pi)  # azimuthal angle
        phi = math.acos(random.uniform(-1, 1))  # polar angle
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        return MPoint(round(x), round(y), round(z))

    # Seed initial two tips:
    #   - one at origin
    #   - one at random point on a circle (2D) or sphere (3D)
    seed1 = MPoint(0, 0, 0)
    if getattr(opts, "use_2d", False):
        seed2 = random_point_on_circle(radius=1.0)
    else:
        seed2 = random_point_on_sphere(radius=1.0)

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

    # Initialise density grid for avoidance behaviours (already 2D)
    grid = DensityGrid(width=100, height=100, resolution=1.0)
    orientator.set_density_grid(grid)

    # Optionally set up anisotropy grid if enabled
    anisotropy_grid = None
    if opts.anisotropy_enabled:
        # In 2D mode, squash the grid into a single layer in z to save memory/compute
        depth = 1 if getattr(opts, "use_2d", False) else 100
        anisotropy_grid = AnisotropyGrid(width=100, height=100, depth=depth, resolution=10.0)
        anisotropy_grid.set_uniform_direction(MPoint(*opts.anisotropy_vector))
        orientator.set_anisotropy_grid(anisotropy_grid)

    # Optionally initialise the antifungal field.
    # This field is kept separate from the density grid: density controls local
    # crowding, whereas drug concentration controls pharmacological inhibition.
    drug_field = None
    if getattr(opts, "drug_field_enabled", False):
        # Build the finite-difference drug field from the Options dataclass.
        drug_field = DrugField2D.from_options(opts)

        # A single explicit initial-condition selector is safer than relying on
        # several booleans at once. The older booleans still work when
        # drug_initial_condition == "legacy", but new test configs should use:
        #   uniform, vertical_sections, or square_perimeter.
        drug_initial_condition = str(getattr(opts, "drug_initial_condition", "legacy")).strip().lower()

        if drug_initial_condition in {"uniform", "background", "none"}:
            # Nothing else to paint: DrugField2D.from_options() already filled
            # the full grid with drug_initial_background_concentration.
            pass

        elif drug_initial_condition == "vertical_sections":
            drug_field.set_vertical_sections(
                x_edges=getattr(opts, "drug_initial_x_edges", []),
                concentrations=getattr(opts, "drug_initial_concentrations", []),
            )

        elif drug_initial_condition == "square_perimeter":
            drug_field.set_square_perimeter_source(
                border_width=getattr(opts, "drug_square_perimeter_width", 10.0),
                source_concentration=getattr(opts, "drug_square_perimeter_concentration", 8.0),
                interior_concentration=getattr(opts, "drug_square_interior_concentration", 0.0),
                maintain_source=getattr(opts, "drug_square_maintain_perimeter", True),
            )

        elif drug_initial_condition == "legacy":
            # Backward-compatible path for configs made before the explicit mode
            # selector existed. If both legacy switches are True, the square
            # perimeter is applied after vertical sections and therefore wins.
            if getattr(opts, "drug_use_vertical_sections", False):
                drug_field.set_vertical_sections(
                    x_edges=getattr(opts, "drug_initial_x_edges", []),
                    concentrations=getattr(opts, "drug_initial_concentrations", []),
                )
            if getattr(opts, "drug_use_square_perimeter", False):
                drug_field.set_square_perimeter_source(
                    border_width=getattr(opts, "drug_square_perimeter_width", 10.0),
                    source_concentration=getattr(opts, "drug_square_perimeter_concentration", 8.0),
                    interior_concentration=getattr(opts, "drug_square_interior_concentration", 0.0),
                    maintain_source=getattr(opts, "drug_square_maintain_perimeter", True),
                )

        else:
            raise ValueError(
                "Unknown drug_initial_condition: "
                f"{drug_initial_condition!r}. Use 'uniform', 'vertical_sections', "
                "'square_perimeter', or 'legacy'."
            )

        # Snapshot the configured starting field before the first diffusion step.
        # This is exported later as drug_field_initial.* when the toggles are on.
        drug_field.capture_initial_concentration()

        # Print a compact sanity check to the console. This makes it immediately
        # obvious whether a supposedly uniform control actually started uniform,
        # instead of discovering the problem only from the final heatmap.
        summary = drug_field.diagnostic_summary(drug_field.initial_concentration)
        print(
            "🧪 Drug field initialised "
            f"mode={drug_initial_condition} "
            f"shape={summary['shape']} "
            f"min={summary['min']:.6g} max={summary['max']:.6g} "
            f"mean={summary['mean']:.6g} centre={summary['centre']:.6g} "
            f"alpha={drug_field.alpha:.6g}"
        )

        # Print a second sanity check for the growth-response logic at the origin,
        # where the seed tip starts. This is especially useful for MIC controls:
        # with the pharmacodynamic model, A == MIC should report raw_growth ≈ 0.
        origin_concentration = drug_field.sample(MPoint(0, 0, 0))
        response_model = str(getattr(opts, "drug_response_model", "pharmacodynamic")).strip().lower()
        if response_model in {"pharmacodynamic", "mic", "mic_pharmacodynamic"}:
            origin_raw_growth = drug_field.pharmacodynamic_growth_rate(
                concentration=origin_concentration,
                mic=getattr(opts, "drug_wildtype_mic", 1.0),
                max_growth_rate=getattr(opts, "growth_rate", 1.0),
                min_growth_rate=getattr(opts, "drug_min_growth_rate", -getattr(opts, "growth_rate", 1.0)),
                hill_coefficient=getattr(opts, "drug_hill_coefficient", 4.0),
            )
            origin_applied_growth = max(origin_raw_growth, 0.0)
        else:
            origin_multiplier = drug_field.growth_multiplier(
                concentration=origin_concentration,
                mic=getattr(opts, "drug_wildtype_mic", 1.0),
                hill_coefficient=getattr(opts, "drug_hill_coefficient", 4.0),
                min_multiplier=getattr(opts, "drug_min_growth_multiplier", 0.0),
            )
            origin_raw_growth = getattr(opts, "growth_rate", 1.0) * origin_multiplier
            origin_applied_growth = max(origin_raw_growth, 0.0)

        print(
            "🧪 Drug response at origin "
            f"model={response_model} "
            f"A={origin_concentration:.6g} "
            f"MIC={getattr(opts, 'drug_wildtype_mic', 1.0):.6g} "
            f"raw_growth={origin_raw_growth:.6g} "
            f"applied_growth={origin_applied_growth:.6g}"
        )

        logger.info(
            "Drug field enabled: mode=%s shape=%s alpha=%.4f",
            drug_initial_condition,
            drug_field.concentration.shape,
            drug_field.alpha,
        )

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
        "anisotropy_grid": anisotropy_grid,
        "drug_field": drug_field,
        "drug_field_history": []
    }


def step_simulation(mycel, components, step):
    """
    Perform one timestep:
        Diffuse antifungal field if enabled,
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
    drug_field = components.get("drug_field", None)

    # Let the antifungal field diffuse before any hyphal tip grows this step.
    # This ordering matches the intended biology: tips respond to the current
    # local concentration after environmental diffusion has occurred.
    if drug_field is not None:
        drug_field.diffuse_once(mycel=mycel, opts=opts)

    # Clear previous field sources and re-add all sections as SectFieldFinders
    aggregator.sources.clear()
    aggregator.add_sections(mycel.get_all_segments(), strength=1.0, decay=1.5)

    # Compute new orientation for each tip using orientator
    use_2d = getattr(opts, "use_2d", False)

    for tip in mycel.get_tips():
        tip.orientation = orientator.compute(tip)

        # In 2D mode, force orientation to lie in the z=0 plane
        if use_2d:
            tip.orientation.coords[2] = 0.0

    # Advance simulation by one time step (grow, branch, prune).
    # The mycelium samples drug_field locally at each active tip when present.
    mycel.step(drug_field=drug_field)

    # If the 2D MP4 is requested, retain one antifungal-field snapshot per
    # simulation step so the saved movie shows diffusion through time rather
    # than only the final concentration map.
    if (
        drug_field is not None
        and getattr(opts, "generate_mycelium_growth_mp4", False)
        and getattr(opts, "use_2d", False)
    ):
        components.setdefault("drug_field_history", []).append(drug_field.concentration.copy())

    # In 2D mode, clamp all segment endpoints to z=0 (safety net)
    if use_2d:
        for seg in mycel.get_all_segments():
            seg.start.coords[2] = 0.0
            seg.end.coords[2] = 0.0

    # Update density grid counts from all segment ends
    grid.update_from_mycel(mycel)

    # Optionally perform nutrient-based kill check if implemented
    if opts.use_nutrient_field and opts.nutrient_repulsion > 0:
        mycel.nutrient_kill_check()

    # Apply any scheduled parameter mutations at this step
    mutator.apply(step, opts)

    # Save a checkpoint if interval reached
    checkpoints.maybe_save(mycel, step)

    # Record stats for plotting later
    stats.update(mycel)

    # Debug-only: compact string summary of current simulation state
    # (off by default; enable with PYCELIUM_LOG_LEVEL=DEBUG)
    logger.debug(str(mycel))


def generate_outputs(mycel, components, output_dir="outputs"):
    """
    Generate artifacts conditionally, based on boolean flags in Options.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Unpack frequently used components
    grid = components["grid"]
    stats = components["stats"]
    opts = components["opts"]
    anisotropy_grid = components.get("anisotropy_grid", None)
    drug_field = components.get("drug_field", None)

        # In 2D mode, prefer 2D visualisations and disable 3D-heavy outputs
    if getattr(opts, "use_2d", False):
        # Ensure 2D outputs are on
        opts.generate_mycelium_2d_png = True
        # Nutrient/aniso 2D defaults – keep whatever the user set, but if missing assume True
        if hasattr(opts, "generate_nutrient_2d_png"):
            opts.generate_nutrient_2d_png = True
        if hasattr(opts, "generate_anisotropy_2d_png"):
            opts.generate_anisotropy_2d_png = True

        # Turn off 3D-only / 3D-heavy outputs
        if hasattr(opts, "generate_mycelium_3d_png"):
            opts.generate_mycelium_3d_png = False
        if hasattr(opts, "generate_nutrient_3d_png"):
            opts.generate_nutrient_3d_png = False
        if hasattr(opts, "generate_anisotropy_3d_png"):
            opts.generate_anisotropy_3d_png = False
        if hasattr(opts, "generate_obj_mesh"):
            opts.generate_obj_mesh = False

    logger.info(f"Saving selected outputs to '{output_dir}'...")

    # --- Core plots ---
    if opts.generate_mycelium_2d_png:
        plot_mycel(mycel, title="2D Projection", save_path=f"{output_dir}/mycelium_2d.png")

    # 3D projections are suppressed in 2D mode
    if opts.generate_mycelium_3d_png:
        plot_mycel_3d(mycel, title="3D Projection", save_path=f"{output_dir}/mycelium_3d.png")

    if opts.generate_mycelium_3d_interactive_html:
        plot_mycel_3d_interactive(mycel, save_path=f"{output_dir}/mycelium_3d_interactive.html")

    # Optional diagnostics
    if opts.generate_density_map_png:
        from vis.density_map import plot_density
        plot_density(grid, save_path=f"{output_dir}/density_map.png")

    if opts.generate_stats_png:
        from vis.analyser import plot_stats
        plot_stats(stats, save_path=f"{output_dir}/stats.png")

    # Nutrient field visuals (only if enabled)
    if opts.use_nutrient_field:
        if opts.generate_nutrient_2d_png:
            plot_nutrient_field_2d(opts, save_path=f"{output_dir}/nutrient_2d.png")
        if opts.generate_nutrient_3d_png:
            plot_nutrient_field_3d(opts, save_path=f"{output_dir}/nutrient_3d.png")

    # Anisotropy visuals (only if enabled)
    if opts.anisotropy_enabled and anisotropy_grid:
        if opts.generate_anisotropy_2d_png:
            plot_anisotropy_2d(anisotropy_grid, save_path=f"{output_dir}/anisotropy_2d.png")
        if opts.generate_anisotropy_3d_png:
            plot_anisotropy_3d(anisotropy_grid, save_path=f"{output_dir}/anisotropy_3d.png")

    # Post-analysis: branching angles (run once; write whichever are enabled)
    if opts.generate_branching_angles_png or opts.generate_branching_angles_csv:
        analyse_branching_angles(
            mycel,
            save_path=(f"{output_dir}/branching_angles.png" if opts.generate_branching_angles_png else None),
            csv_path=(f"{output_dir}/branching_angles.csv" if opts.generate_branching_angles_csv else None),
        )

    # (Optional) Tip orientations
    if opts.generate_tip_orientations_png or opts.generate_tip_orientations_csv:
        analyse_tip_orientations(
            mycel,
            save_path=(f"{output_dir}/tip_orientations.png" if opts.generate_tip_orientations_png else None),
            csv_path=(f"{output_dir}/orientations.csv" if opts.generate_tip_orientations_csv else None),
        )

    # Final state & histories
    if opts.generate_mycelium_final_csv:
        export_to_csv(mycel, f"{output_dir}/mycelium_final.csv", all_time=False)

    if opts.generate_density_map_csv:
        export_grid_to_csv(grid, f"{output_dir}/density_map.csv")

    # Optional combined visualisation: final mycelium over the final drug field.
    # This keeps mycelium_2d.png and drug_field_final.png as separate outputs,
    # but adds a contextual figure showing their interaction.
    drug_plot_vmax = getattr(opts, "drug_plot_max_concentration", None)
    if drug_field is not None and getattr(opts, "generate_mycelium_drug_overlay_png", False):
        plot_mycelium_on_drug_field(
            mycel,
            drug_field,
            save_path=f"{output_dir}/mycelium_drug_overlay.png",
            title="Mycelium on antifungal field",
            vmax=drug_plot_vmax,
        )

    # Antifungal field exports. These are only written when the drug field is
    # enabled and the corresponding output toggles are True. Initial-field
    # exports are especially useful for control tests because the final heatmap
    # has already changed through diffusion.
    if drug_field is not None:
        final_summary = drug_field.diagnostic_summary()
        print(
            "🧪 Drug field final "
            f"min={final_summary['min']:.6g} max={final_summary['max']:.6g} "
            f"mean={final_summary['mean']:.6g} centre={final_summary['centre']:.6g}"
        )

        initial = getattr(drug_field, "initial_concentration", None)
        if initial is not None:
            if getattr(opts, "generate_drug_field_initial_npy", False):
                drug_field.export_npy(f"{output_dir}/drug_field_initial.npy", array=initial)
            if getattr(opts, "generate_drug_field_initial_csv", False):
                drug_field.export_csv(f"{output_dir}/drug_field_initial.csv", array=initial)
            if getattr(opts, "generate_drug_field_initial_png", False):
                drug_field.export_png(
                    f"{output_dir}/drug_field_initial.png",
                    array=initial,
                    title="Initial antifungal field",
                    vmax=drug_plot_vmax,
                )

        if getattr(opts, "generate_drug_field_npy", False):
            drug_field.export_npy(f"{output_dir}/drug_field_final.npy")
        if getattr(opts, "generate_drug_field_csv", False):
            drug_field.export_csv(f"{output_dir}/drug_field_final.csv")
        if getattr(opts, "generate_drug_field_png", False):
            drug_field.export_png(f"{output_dir}/drug_field_final.png", vmax=drug_plot_vmax)

        # Optional biofilm/barrier diagnostics for biomass-hindered diffusion.
        # These are useful for checking whether dense hyphae are actually making
        # the colony interior less permeable to antifungal diffusion.
        if getattr(opts, "drug_diffusion_hindered_by_biomass", False):
            biomass_barrier = getattr(drug_field, "last_biomass_barrier", None)
            effective_diffusion = getattr(drug_field, "last_effective_diffusion", None)

            if biomass_barrier is not None:
                if getattr(opts, "generate_drug_biomass_barrier_npy", False):
                    drug_field.export_npy(f"{output_dir}/drug_biomass_barrier_final.npy", array=biomass_barrier)
                if getattr(opts, "generate_drug_biomass_barrier_csv", False):
                    drug_field.export_csv(f"{output_dir}/drug_biomass_barrier_final.csv", array=biomass_barrier)
                if getattr(opts, "generate_drug_biomass_barrier_png", False):
                    drug_field.export_png(
                        f"{output_dir}/drug_biomass_barrier_final.png",
                        array=biomass_barrier,
                        title="Final hyphal biomass diffusion barrier",
                        colorbar_label="Dimensionless hyphal biomass barrier",
                    )

            if effective_diffusion is not None:
                if getattr(opts, "generate_drug_effective_diffusion_npy", False):
                    drug_field.export_npy(f"{output_dir}/drug_effective_diffusion_final.npy", array=effective_diffusion)
                if getattr(opts, "generate_drug_effective_diffusion_csv", False):
                    drug_field.export_csv(f"{output_dir}/drug_effective_diffusion_final.csv", array=effective_diffusion)
                if getattr(opts, "generate_drug_effective_diffusion_png", False):
                    drug_field.export_png(
                        f"{output_dir}/drug_effective_diffusion_final.png",
                        array=effective_diffusion,
                        title="Final effective antifungal diffusion coefficient",
                        colorbar_label="Effective diffusion coefficient",
                    )

    # Time-series CSV + animation (dependency handled)
    series_path = f"{output_dir}/mycelium_time_series.csv"
    need_series_for_mp4 = opts.generate_mycelium_growth_mp4

    if opts.generate_mycelium_time_series_csv or need_series_for_mp4:
        export_tip_history(mycel, series_path)

    if opts.generate_mycelium_growth_mp4:
        if getattr(opts, "use_2d", False):
            animate_network_growth_2d(
                network_history=getattr(mycel, "network_time_series", []),
                save_path=f"{output_dir}/mycelium_growth_2d.mp4",
                interval=getattr(opts, "mycelium_growth_mp4_interval_ms", 100),
                drug_field=drug_field,
                drug_field_history=components.get("drug_field_history", []),
                vmax=drug_plot_vmax,
                dpi=getattr(opts, "mycelium_growth_mp4_dpi", 150),
                show_colorbar=getattr(opts, "mycelium_growth_mp4_show_drug_colorbar", True),
                highlight_zero_growth=getattr(opts, "highlight_zero_growth_hyphae", True),
                zero_growth_color=getattr(opts, "zero_growth_highlight_color", (1.0, 0.15, 0.0)),
                zero_growth_linewidth=getattr(opts, "zero_growth_highlight_linewidth", 2.6),
                zero_growth_marker_size=getattr(opts, "zero_growth_highlight_marker_size", 34.0),
            )
        else:
            animate_growth(
                csv_path=series_path,
                save_path=f"{output_dir}/mycelium_growth.mp4",
                interval=getattr(opts, "mycelium_growth_mp4_interval_ms", 100),
            )

        if not opts.generate_mycelium_time_series_csv:
            try:
                os.remove(series_path)
            except OSError:
                pass

    if opts.generate_biomass_and_tips_history:
        export_biomass_history(mycel, f"{output_dir}/biomass_and_tips_history.csv")

    # 3D mesh (OBJ) if desired
    if opts.generate_obj_mesh:
        export_to_obj(mycel, f"{output_dir}/mycelium.obj")

def simulate(opts, steps=120):
    """
    Top-level function to run a full sim loop, handle autostop, and then call generate_outputs at the end.
    """
    # Initialise sim and components
    mycel, components = setup_simulation(opts)

    # Rate-limited heartbeat (visible when PYCELIUM_LOG_LEVEL=INFO)
    log_every = parse_int_env("PYCELIUM_LOG_EVERY", 50)

    try:
        # Loop for the requested no. steps
        for step in range(steps):
            step_simulation(mycel, components, step)

            # Heartbeat only every N steps; keep it lightweight
            if log_every > 0 and (step % log_every) == 0:
                logger.info(f"step {step} | tips={len(mycel.get_tips())} | sections={len(mycel.get_all_segments())}")

            # Check AutoStop condition
            if components["autostop"].check(mycel, step):
                logger.info("AutoStop triggered; terminating early.")
                break

    except KeyboardInterrupt:
        # Allow user to interrupt simulation with Ctrl+C and still save results
        logger.warning("Interrupted by user. Saving final state...")

    # Determine final output folder (env takes precedence in all modes)
    output_dir = os.getenv("BATCH_OUTPUT_DIR", "outputs")
    logger.info(f"Saving outputs to: {output_dir}")
    generate_outputs(mycel, components, output_dir=output_dir)  # Generate all plots and exports

    print("✅ Simulation completed")

if __name__ == "__main__":
    # If run directly, load a default config and simulate
    opts = load_options_from_json("configs/example.json")
    simulate(opts, steps=120)
