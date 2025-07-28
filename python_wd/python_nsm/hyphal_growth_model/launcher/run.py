# launcher/run.py

import sys
import os
import argparse
from config.sim_config import load_options_from_json, load_options_from_cli
from gui.sim_gui import OptionGUI
from main import simulate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_gui():
    print("🖥️ Launching GUI mode...")
    OptionGUI()

def run_cli(config=None, steps=None):
    print("🧬 Running standard simulation...")
    if config:
        opts = load_options_from_json(config)
    else:
        opts, steps = load_options_from_cli()
    simulate(opts, steps or 120)

def run_sweep(param, values, steps):
    print(f"🧪 Running parameter sweep on '{param}'")
    results = run_parameter_sweep(param, values, steps)
    plot_sweep(results, param_label=param)

def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Hyphal Growth Simulator")
    parser.add_argument("--mode", type=str, default="gui", choices=["gui", "cli", "sweep"], help="Launch mode")
    parser.add_argument("--config", type=str, help="Path to JSON config")
    parser.add_argument("--steps", type=int, help="Override number of steps")
    parser.add_argument("--sweep_param", type=str, help="Param to sweep (e.g. branch_probability)")
    parser.add_argument("--sweep_values", nargs="+", type=float, help="Values to sweep (e.g. 0.1 0.2 0.3)")
    parser.add_argument("--sweep_steps", type=int, default=120, help="Steps per sweep simulation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "gui":
        run_gui()

    elif args.mode == "cli":
        run_cli(config=args.config, steps=args.steps)

    elif args.mode == "sweep":
        if not args.sweep_param or not args.sweep_values:
            print("⚠️ Please specify --sweep_param and --sweep_values for sweep mode.")
        else:
            run_sweep(args.sweep_param, args.sweep_values, args.sweep_steps)
