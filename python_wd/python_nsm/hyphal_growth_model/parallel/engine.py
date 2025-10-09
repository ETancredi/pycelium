# parallel/engine.py

"""
Deterministic engine: exact transplant of main.step_simulation()
around Mycel.step(), without delegating to main.py.

Sequence (matches your main.step_simulation):
  1) aggregator.sources.clear()
  2) aggregator.add_sections(mycel.get_all_segments(), strength=1.0, decay=1.5)
  3) For each tip: tip.orientation = orientator.compute(tip)
  4) mycel.step()        # grow, destructors, branching, snapshot, time, prune, histories
  5) grid.update_from_mycel(mycel)
  6) if opts.use_nutrient_field and opts.nutrient_repulsion > 0: mycel.nutrient_kill_check()
  7) mutator.apply(step, opts)
  8) checkpoints.maybe_save(mycel, step)
  9) stats.update(mycel)
 10) logger.debug(str(mycel))

No additional RNG use is introduced; RNG calls occur only where your code
already calls them (Section.maybe_branch + pruning inside mycel.step()).
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import logging

from core.options import Options

logger = logging.getLogger("pycelium.parallel.engine")

class ParallelStepEngine:
    def __init__(self, apply_order: str = "id", workers: int = 0):
        self.apply_order = apply_order
        self.workers = int(workers)

    def step_parallel_equivalent(
        self,
        mycel,
        components: Dict[str, Any],
        step: int,
        master_seed: Optional[int],   # kept for future phases; unused here
    ) -> None:
        """
        Perform one step, reproducing main.step_simulation() exactly (no delegation).
        """

        # Unpack components (same names as in your main.py)
        aggregator   = components["aggregator"]
        grid         = components["grid"]
        orientator   = components["orientator"]
        checkpoints  = components["checkpoints"]
        autostop     = components["autostop"]      # unused here but kept for parity
        mutator      = components["mutator"]
        stats        = components["stats"]
        opts: Options = components["opts"]

        # 1) Refresh field sources
        aggregator.sources.clear()
        aggregator.add_sections(mycel.get_all_segments(), strength=1.0, decay=1.5)

        # 2) Update tip orientations using orientator
        for tip in mycel.get_tips():
            tip.orientation = orientator.compute(tip)

        # 3) Advance simulation core step (your Mycel.step implementation)
        mycel.step()

        # 4) Update density grid from current mycel state
        grid.update_from_mycel(mycel)

        # 5) Optional nutrient-based kill check (if implemented/enabled)
        if getattr(opts, "use_nutrient_field", False) and getattr(opts, "nutrient_repulsion", 0) > 0:
            # Guard: only call if method exists on Mycel, as in your main.py comment
            if hasattr(mycel, "nutrient_kill_check"):
                mycel.nutrient_kill_check()

        # 6) Apply runtime parameter mutations
        mutator.apply(step, opts)

        # 7) Checkpoint at interval
        checkpoints.maybe_save(mycel, step)

        # 8) Update stats
        stats.update(mycel)

        # 9) Debug summary (same as main.py)
        logger.debug(str(mycel))
