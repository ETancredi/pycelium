# tests/run_parallel_hash.py

import os
import sys
import csv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.sim_config import load_options_from_json
from tests.determinism_harness import mycel_hash
import main as runner
from parallel.engine import ParallelStepEngine

def run_and_hash(opts, steps, output_csv="outputs/state_hashes_parallel.csv"):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    mycel, components = runner.setup_simulation(opts)
    engine = ParallelStepEngine(workers=0)
    rows = [("step", "sha256")]
    seed = getattr(opts, "seed", 123)
    for step in range(steps):
        engine.step_parallel_equivalent(mycel, components, step, master_seed=seed)
        rows.append((step, mycel_hash(mycel)))
    rows.append(("final", mycel_hash(mycel)))

    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)

    for r in rows:
        print(f"{r[0]},{r[1]}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("config_json", nargs="?", default="config/param_config.json")
    p.add_argument("steps", type=int, nargs="?", default=60)
    p.add_argument("--output", default="outputs/state_hashes_parallel.csv")
    args = p.parse_args()

    opts = load_options_from_json(args.config_json)
    run_and_hash(opts, args.steps, args.output)
