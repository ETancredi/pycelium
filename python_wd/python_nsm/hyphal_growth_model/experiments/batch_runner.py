# batch_runner.py

# Imports
import sys # Access to Python runtime environment
import os # Operating system interfaces 
import json # JSON serialisation/deserialisation for config files
import csv # CSV reading/writing for summary output
import shutil # High-level file operations (copying files)
import random # Random number generation for seed assignment
from datetime import datetime # Date/time for timestamping batch runs

# Multiprocessing for parallel execution
from multiprocessing import Pool 
from functools import partial # Fix arguments for pool worker function

# Ensure project root is on the import path so we can import main.simulate
# __file__is this scripts's path; we go up one directory and add to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core simulation entry point and options class
from main import simulate # simulate (opts: Options, steps: int) runs on simulation
from config.sim_config import Options # Options(dataclass) for sim params


def load_config(config_path):
    """
    Load a JSON config from disk.
    Args:
        config_path (str): Path to .json file containing batch configuration.
    Returns:
        dict: Parsed JSON object.
    """
    with open(config_path, "r") as f:
        return json.load(f)

def run_batch(config_path):
    """
    Run a series of simulations sequentially based on a batch config.
    Args:
        config_path (str): Path to JSON file with structure:
            {
              "runs": [
                { "name": "run1", "options": {...}, "steps": },
                ...
              ]
            }
    """
    # Load and parse the user-provided batch configuration
    batch_config = load_config(config_path)
    # Create timestamped folder to store all outputs for this batch
    batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_folder = os.path.join("batch_outputs", batch_name)
    os.makedirs(batch_folder, exist_ok=True) # Make directory, skip if exists
    print(f"📦 Launching batch: {batch_name}")

    # Archive the config used for reproducibility
    shutil.copy(config_path, os.path.join(batch_folder, "batch_config_used.json"))
    print(f"🗄 Saved batch config to {batch_folder}/batch_config_used.json")
    
    summary_data = [] # Collect pre-run status infor for summary csv

    # Iterate over each run configuration in the batch
    for i, run_cfg in enumerate(batch_config["runs"]):
        # Determine a human-friendly run name
        run_name = run_cfg.get("name", f"run_{i+1}")
        # Instantiate simulation Options from the run's "options" dict
        opts = Options(**run_cfg["options"])
        # Number of simulation steps; default to 120 if not specified
        steps = run_cfg.get("steps", 120)
        # Create a unique subfolder for this sim run using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_folder_name = f"{run_name}_{i+1:03d}_{timestamp}"
        output_dir = os.path.join(batch_folder, sim_folder_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"▶️ Running: {run_name} (#{i+1}) → {output_dir}")

        try:
            # Inform downstream code where to save outputs
            os.environ["BATCH_OUTPUT_DIR"] = output_dir
            # Run the sim
            simulate(opts, steps)

            # If successful, record status and paths
            summary_data.append({
                "run_name": run_name,
                "steps": steps,
                "status": "✅ Success",
                "output_dir": output_dir
            })

        except Exception as e:
            # On error, log exception message and record failure
            print(f"❌ Error in {run_name}: {e}")
            summary_data.append({
                "run_name": run_name,
                "steps": steps,
                "status": f"❌ Failed: {e}",
                "output_dir": output_dir
            })

    # After all runs, write a summary CSV in the batch folder
    summary_file = os.path.join(batch_folder, "batch_summary.csv")
    with open(summary_file, "w", newline="") as f:
        # Use keys from the first summary entry as CSV columns
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"\n✅ All simulations complete. Summary saved to {summary_file}")


def worker(run_cfg, batch_folder):
    """
    Worker function for parallel execution of a single run.
    Args:
        run_cfg (dict): Single run config with keys "name", "options", "steps", and "seed".
        batch_folder (str): Base folder for writing outputs.
    Returns:
        dict: Summary info for this run (same shape as run_batch's summary_data entries).
    """
    # Extract run name and instantiate Options
    run_name = run_cfg.get("name", "unnamed")
    opts = Options(**run_cfg["options"])
    steps = run_cfg.get("steps", 120)
    seed = run_cfg.get("seed") # Seed is injected propr to pooling

    # Create per-run output directory with seed in its name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_folder_name = f"{run_name}_seed{seed}_{timestamp}"
    output_dir = os.path.join(batch_folder, sim_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Export output directory for simulate() to use
    os.environ["BATCH_OUTPUT_DIR"] = output_dir

    try:
        print(f"▶️ [Worker] Running: {run_name} (seed={seed}) → {output_dir}")
        simulate(opts, steps)
        return {
            "run_name": run_name,
            "steps": steps,
            "status": "✅ Success",
            "output_dir": output_dir,
            "seed": seed
        }
    except Exception as e:
        print(f"❌ Error in {run_name}: {e}")
        return {
            "run_name": run_name,
            "steps": steps,
            "status": f"❌ Failed: {e}",
            "output_dir": output_dir,
            "seed": seed
        }


def run_batch_parallel(config_path):
    """
    Run batch simulations in parallel, with optional replicates and random seeds.
    Args:
        config_path (str): Path to JSON batch confi with keys "run" and optional "replicates".
    """
    # Determine the no. CPU cores and reserve one for OS
    num_cores = os.cpu_count()
    processes = max(1, num_cores - 1)
    print(f"🖥️ Detected {num_cores} cores; using {processes} workers.")

    batch_config = load_config(config_path)
    # Create a folder for this parallel batch
    batch_name = f"batch_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_folder = os.path.join("batch_outputs", batch_name)
    os.makedirs(batch_folder, exist_ok=True)
    print(f"📦 Launching parallel batch: {batch_name}")

    # Prepare expanded runs with random seeds
    all_runs = []
    used_seeds = set()
    replicates = batch_config.get("replicates", 1)

    for i, run_cfg in enumerate(batch_config["runs"]):
        # Create multiple replicates if requested
        for r in range(replicates):
            # Deep copy the run config so modifications don't leak
            run = json.loads(json.dumps(run_cfg))
            run_name = run_cfg.get("name", f"run_{i+1}")
            run["name"] = f"{run_name}_rep{r+1}"

            # Generate a unique random seed from 1-1000000
            while True:
                seed = random.randint(1, 1_000_000)
                if seed not in used_seeds:
                    used_seeds.add(seed)
                    break

            # Inject seed into both options and summary metadata
            run["options"]["seed"] = seed
            run["seed"] = seed
            all_runs.append(run)

    # Save full batch config (including replicates + seeds)
    with open(os.path.join(batch_folder, "batch_config_used.json"), "w") as f:
        json.dump({"runs": all_runs}, f, indent=2)
    print(f"🗄 Saved expanded config with seeds to: batch_config_used.json")

    # Launch pool
    with Pool(processes=processes) as pool:
        # Partial binds batch_folder to each worker call
        results = pool.map(partial(worker, batch_folder=batch_folder), all_runs)

    # Write summary of parallel runs to CSV
    summary_file = os.path.join(batch_folder, "batch_summary.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ All simulations complete. Summary saved to {summary_file}")


if __name__ == "__main__":
    # Default config for CLI invocation
    config_path = "config/batch_config.json"
    # To use parallel: uncomment below and comment out run_batch
    run_batch_parallel(config_path)
    # To run sequentially instead, comment the line above and uncomment:
    # run_batch(config_path)
