# tests/test_parallel_skeleton_scaffolding.py
"""
Sanity test to ensure the parallel scaffold is behaviour-identical *before* we
transplant logic. We reuse the harness hasher to compare against a direct CSF run.

Key points:
- Re-seed RNGs for each run
- Reload fresh Options for each run (avoid mutation bleed)
- Reset the global Section ID counter between runs
"""

import os
import sys
import random
import numpy as np
import itertools

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.sim_config import load_options_from_json
from tests.determinism_harness import mycel_hash
import main as runner
from parallel.engine import ParallelStepEngine
from core import section as section_mod  # to reset _SECTION_ID_GEN

def _effective_seed(opts):
    s = getattr(opts, "seed", None)
    return 0 if s is None else int(s)

def _seed_globals(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def _reset_section_id_counter():
    # Reset the global counter so a new simulation starts from the same ID sequence
    section_mod._SECTION_ID_GEN = itertools.count()

def run_direct(opts_path, steps):
    # Independent Options instance + deterministic seeding
    opts = load_options_from_json(opts_path)
    _seed_globals(_effective_seed(opts))
    _reset_section_id_counter()
    mycel, components = runner.setup_simulation(opts)
    hashes = []
    for step in range(steps):
        runner.step_simulation(mycel, components, step)
        hashes.append(mycel_hash(mycel))
    return hashes

def run_scaffold(opts_path, steps):
    # Independent Options instance + deterministic seeding
    opts = load_options_from_json(opts_path)
    seed = _effective_seed(opts)
    _seed_globals(seed)
    _reset_section_id_counter()
    mycel, components = runner.setup_simulation(opts)
    engine = ParallelStepEngine(workers=0)  # serial proposals for now
    hashes = []
    for step in range(steps):
        engine.step_parallel_equivalent(mycel, components, step, master_seed=seed)
        hashes.append(mycel_hash(mycel))
    return hashes

if __name__ == "__main__":
    cfg = "config/param_config.json"
    steps = 20
    h1 = run_direct(cfg, steps)
    h2 = run_scaffold(cfg, steps)
    for i, (a, b) in enumerate(zip(h1, h2)):
        print(i, a, b, "OK" if a == b else "MISMATCH")
