# tests/test_parallel_skeleton_scaffolding.py
"""
Sanity test to ensure the parallel scaffold is behaviour-identical *before* we
transplant logic. We reuse the harness hasher to compare against a direct CSF run.
"""

import os
import sys
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.sim_config import load_options_from_json
from tests.determinism_harness import mycel_hash
import main as runner
from parallel.engine import ParallelStepEngine

def _effective_seed(opts):
    s = getattr(opts, "seed", None)
    return 0 if s is None else int(s)

def _seed_globals(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def run_direct(opts, steps):
    # ensure deterministic init even if opts.seed is None
    _seed_globals(_effective_seed(opts))
    mycel, components = runner.setup_simulation(opts)
    hashes = []
    for step in range(steps):
        runner.step_simulation(mycel, components, step)
        hashes.append(mycel_hash(mycel))
    return hashes

def run_scaffold(opts, steps, seed):
    # ensure deterministic init even if opts.seed is None
    _seed_globals(_effective_seed(opts))
    mycel, components = runner.setup_simulation(opts)
    engine = ParallelStepEngine(workers=0)  # serial proposals for now
    hashes = []
    for step in range(steps):
        engine.step_parallel_equivalent(mycel, components, step, master_seed=seed)
        hashes.append(mycel_hash(mycel))
    return hashes

if __name__ == "__main__":
    opts = load_options_from_json("config/param_config.json")
    steps = 20
    seed = _effective_seed(opts)
    h1 = run_direct(opts, steps)
    h2 = run_scaffold(opts, steps, seed)
    for i, (a, b) in enumerate(zip(h1, h2)):
        print(i, a, b, "OK" if a == b else "MISMATCH")
