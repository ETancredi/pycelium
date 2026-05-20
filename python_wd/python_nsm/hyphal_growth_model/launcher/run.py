# launcher/run.py

# Imports
import argparse  # Parsing command-line args
from config.sim_config import load_options_from_json, load_options_from_cli  # Helper functions to load Options from JSON or from CLI flags
from gui.sim_gui import OptionGUI  # GUI class to launch graphical simulator
from main import simulate  # Run the simulation headlessly
import os
import random # For unified seed
import numpy as np
from io_utils.run_paths import resolve_seed, compute_run_dir # Per-run folder helpers


def run_gui():
    """
    Launch the simulator in GUI mode.
    Initialises and displays the Tkinter interface.
    """
    print("🖥️ Launching GUI mode...")
    OptionGUI()  # Instantiate the GUI, which calls mainloop internally


def run_cli(config=None, steps=None):
    """
    Run the simulator in CLI (command-line) mode.
    Args:
        config (str or None): Path to a JSON file defining Options
        steps (int or None): No. sim steps to override.
    """
    print("🧬 Running standard simulation...")
    if config:
        # Load sim opts from the provided JSON config file
        opts = load_options_from_json(config)
    else:
        # Load opts and steps from CLI flags (prompts user or uses defaults)
        opts, steps = load_options_from_cli()

    # --- NEW: resolve seed & set RNGs consistently across CLI/GUI/Batch ---
    seed = resolve_seed(getattr(opts, "seed", None))
    opts.seed = seed
    random.seed(seed)
    np.random.seed(seed)

    # --- NEW: create unique per-run folder & inform downstream via env var ---
    run_dir = compute_run_dir("outputs", seed)  # e.g., outputs/20250925_1_123456789
    os.environ["BATCH_OUTPUT_DIR"] = str(run_dir)
    print(f"📂 Outputs will be written to: {run_dir} (seed={seed})")

    # Run the sim with loaded opts and steps (default 120)
    simulate(opts, steps or 120)


def run_sweep(param, values, steps):
    """
    Perform a param sweep: run multiple sims varying one param.
    Args:
        param (str): Name of Options param to sweep.
        values: (list of float): Values to assign to that param.
        Steps (int): No. steps for each simulation.
    """
    print(f"🧪 Running parameter sweep on '{param}'")
    # Execute sweep logic and collect results
    results = run_parameter_sweep(param, values, steps)
    # Plot results using a helper function
    plot_sweep(results, param_label=param)


def parse_args():
    """
    Define and parse command-line args for this launcher.
    Returns:
        Namespace: Parsed args w/ attributes:
            mode, config, steps, sweep_param, sweep_values, sweep_steps
    """
    parser = argparse.ArgumentParser(description="Launch the Hyphal Growth Simulator")
    # Mode selection: gui, cli, or sweep
    parser.add_argument("--mode", type=str, default="gui", choices=["gui", "cli", "sweep"], help="Launch mode")
    # Optional path to JSON config for cli mode
    parser.add_argument("--config", type=str, help="Path to JSON config")
    # Optional override for no. steps
    parser.add_argument("--steps", type=int, help="Override number of steps")
    # For sweep mode: name of param to vary
    parser.add_argument("--sweep_param", type=str, help="Param to sweep (e.g. branch_probability)")
    # For sweep mode: list of values to test
    parser.add_argument("--sweep_values", nargs="+", type=float, help="Values to sweep (e.g. 0.1 0.2 0.3)")
    # For sweep mode: no. steps per sim
    parser.add_argument("--sweep_steps", type=int, default=120, help="Steps per sweep simulation")
    return parser.parse_args()


if __name__ == "__main__":
    # Entry point when this script is executed directly
    args = parse_args()

    if args.mode == "gui":
        # GUI mode: ignore other flags
        run_gui()

    elif args.mode == "cli":
        # CLI mode: pass config path and steps override
        run_cli(config=args.config, steps=args.steps)

    elif args.mode == "sweep":
        # Sweep mode: require both parameter name and values
        if not args.sweep_param or not args.sweep_values:
            print("⚠️ Please specify --sweep_param and --sweep_values for sweep mode.")
        else:
            run_sweep(args.sweep_param, args.sweep_values, args.sweep_steps)
