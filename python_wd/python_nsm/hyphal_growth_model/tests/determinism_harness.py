
# tests/determinism_harness.py

"""
Determinism harness for Pycelium (CSF baseline).
- Runs the single-thread engine step-by-step.
- Emits a canonical SHA-256 hash of the full simulation state after each step.
- Writes a CSV with (step, sha256).
This file intentionally does **not** modify core code.
"""

import os
import csv
import sys
import json
import math
import struct
import hashlib
from typing import Iterable, Tuple

# Import from project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.point import MPoint
from core.section import Section
from core.options import Options
from core import section as section_mod
import main as runner  # setup_simulation, step_simulation

# --- Canonical snapshot encoding ---

def _bool_byte(x: bool) -> bytes:
    return b"\x01" if x else b"\x00"

def _pack_f64(x: float) -> bytes:
    # Normalise -0.0/+0.0 and NaNs to a canonical bit-pattern
    if x == 0.0:
        x = 0.0  # strip -0.0
    # For NaN, use a fixed payload
    if isinstance(x, float) and math.isnan(x):
        return struct.pack("<Q", 0x7ff8000000000000)
    return struct.pack("<d", float(x))

def _pack_i64(x: int) -> bytes:
    return struct.pack("<q", int(x))

def section_bytes(sec: Section) -> bytes:
    """
    Deterministic byte representation of a Section.
    Order and field set must remain stable.
    """
    parent_id = -1 if sec.parent is None else int(sec.parent.id)
    parts = []
    parts.append(_pack_i64(int(sec.id)))
    parts.append(_pack_i64(parent_id))
    # Start / End coordinates
    parts.append(_pack_f64(sec.start.x)); parts.append(_pack_f64(sec.start.y)); parts.append(_pack_f64(sec.start.z))
    parts.append(_pack_f64(sec.end.x));   parts.append(_pack_f64(sec.end.y));   parts.append(_pack_f64(sec.end.z))
    # Orientation & memory
    parts.append(_pack_f64(sec.orientation.x)); parts.append(_pack_f64(sec.orientation.y)); parts.append(_pack_f64(sec.orientation.z))
    if hasattr(sec, "direction_memory"):
        parts.append(_pack_f64(sec.direction_memory.x)); parts.append(_pack_f64(sec.direction_memory.y)); parts.append(_pack_f64(sec.direction_memory.z))
    # Scalars
    for name in ["length", "age"]:
        parts.append(_pack_f64(getattr(sec, name, 0.0)))
    # Flags
    for name in ["is_tip", "is_dead"]:
        parts.append(_bool_byte(bool(getattr(sec, name, False))))
    # Colour (RGB as floats 0..1)
    r,g,b = getattr(sec, "color", (0.0,0.0,0.0))
    parts.append(_pack_f64(r)); parts.append(_pack_f64(g)); parts.append(_pack_f64(b))
    # Branch counts / metadata if present
    for name in ["children_count", "branch_count"]:
        if hasattr(sec, name):
            parts.append(_pack_i64(int(getattr(sec, name))))
    # Bounds clamping indicator if available
    if hasattr(sec, "clamped_last_step"):
        parts.append(_bool_byte(bool(sec.clamped_last_step)))
    return b"".join(parts)

def mycel_bytes(mycel) -> bytes:
    """
    Deterministic byte representation of the whole Mycel state.
    Iterates sections in ascending ID.
    """
    parts = []
    # Global header
    parts.append(_pack_f64(float(getattr(mycel, "time", 0.0))))
    # Sections (sorted by id)
    sections = list(getattr(mycel, "sections", []))
    sections.sort(key=lambda s: int(s.id))
    parts.append(_pack_i64(len(sections)))
    for sec in sections:
        parts.append(section_bytes(sec))
    return b"|".join(parts)

def mycel_hash(mycel) -> str:
    return hashlib.sha256(mycel_bytes(mycel)).hexdigest()

def run_and_hash(opts: Options, steps: int, output_csv: str = "outputs/state_hashes.csv") -> None:
    """
    Run the sim step-by-step, recording a SHA-256 hash after each step.
    """
    # Ensure output dir
    out_dir = os.path.dirname(output_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Build sim
    mycel, components = runner.setup_simulation(opts)

    rows = [("step","sha256")]
    for step in range(steps):
        # advance sim one step
        runner.step_simulation(mycel, components, step)
        # hash state
        rows.append((step, mycel_hash(mycel)))

    # also hash final state one more time with explicit "final" label
    rows.append(("final", mycel_hash(mycel)))

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)

    # Print to stdout for convenience
    for r in rows:
        print(f"{r[0]},{r[1]}")

if __name__ == "__main__":
    # CLI usage:
    #   python -m tests.determinism_harness configs/param_config.json 60
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("config_json", nargs="?", default="config/param_config.json")
    parser.add_argument("steps", type=int, nargs="?", default=60)
    parser.add_argument("--output", default="outputs/state_hashes.csv")
    args = parser.parse_args()

    # Load Options via project helper if available; else parse minimal JSON to Options
    from config.sim_config import load_options_from_json
    opts = load_options_from_json(args.config_json)
    run_and_hash(opts, args.steps, args.output)
