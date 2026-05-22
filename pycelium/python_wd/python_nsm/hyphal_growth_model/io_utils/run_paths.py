# io_utils/run_paths.py
#
# Purpose:
#   Single source of truth for:
#     1) Resolving the RNG seed (use config.seed if provided; otherwise create one).
#     2) Creating a unique per-run output directory in the format:
#            outputs/YYYYMMDD_jobID_seed
#        where:
#            - YYYYMMDD is in UTC (stable, machine-agnostic),
#            - jobID is a simple, monotonically-increasing integer scoped to the outputs folder,
#            - seed is the integer RNG seed used for this run.
#
# Rationale:
#   - GUI/CLI runs previously overwrote "outputs/". We fix that by computing a run_dir here.
#   - Batch mode already sets BATCH_OUTPUT_DIR and is left as-is. 
#   - main.py respects BATCH_OUTPUT_DIR across checkpoints + exports, so once launchers set
#     that env var, everything (plots, CSVs, OBJ, animations) will flow into run_dir. 

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import random
import re

# Detect folders we created to compute the next job id.
# Example folder names we match: "20250925_1_123456789"
FOLDER_PATTERN = re.compile(r"^\d{8}_(\d+)_\d+$")


def _utc_yyyymmdd() -> str:
    """
    Return today's date in UTC formatted as YYYYMMDD.
    UTC is chosen to avoid surprises when running across machines/timezones.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _next_job_id(outputs_dir: Path) -> int:
    """
    Scan outputs_dir for previously created run folders and return the next job id.
    Starts at 1 when none exist.

    We look only for folders that match our pattern to avoid picking up unrelated dirs.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue
        m = FOLDER_PATTERN.match(child.name)
        if m:
            try:
                jid = int(m.group(1))
                if jid > max_id:
                    max_id = jid
            except ValueError:
                # defensive: ignore weird names
                pass
    return max_id + 1


def resolve_seed(config_seed: int | None) -> int:
    """
    Determine the RNG seed for this run.

    Rules:
      - If the Options/config provides a seed, use it verbatim.
      - Otherwise generate a 31-bit random seed (friendly for numpy/random).

    Returns:
      int: the seed to use for random and numpy seeding.
    """
    if config_seed is not None:
        return int(config_seed)
    # 31-bit int (1 .. 2^31-1), avoids signed/unsigned quirks.
    return random.randint(1, 2**31 - 1)


def compute_run_dir(
    outputs_root: str | Path,
    seed: int,
    job_id: int | None = None,
    zero_pad: int | None = None,
) -> Path:
    """
    Create and return the per-run output directory:
        <outputs_root>/<YYYYMMDD>_<jobID>_<seed>

    Args:
      outputs_root: base outputs folder (e.g., "outputs")
      seed: integer RNG seed (included in folder name)
      job_id: provide to override the auto-increment id (mostly useful for tests)
      zero_pad: if provided, zero-pads jobID to this width (e.g., 4 -> 0001)

    Returns:
      Path: the created run directory.
    """
    outputs_root = Path(outputs_root)
    ymd = _utc_yyyymmdd()
    jid = job_id if job_id is not None else _next_job_id(outputs_root)
    jid_str = f"{jid:0{zero_pad}d}" if zero_pad else str(jid)
    run_dir = outputs_root / f"{ymd}_{jid_str}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
