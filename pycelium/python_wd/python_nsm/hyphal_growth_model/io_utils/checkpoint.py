# io_utils/checkpoint.py

# Imports
import time 
import logging
logger = logging.getLogger("pycelium")
from io_utils.saver import save_to_json # Serialise Mycel object to JSON
from pathlib import Path # Path object for filesystem path manipulations

class CheckpointSaver:
    """
    Periodically writes out the full simulation state to JSON files
    Useful for resuming long runs or inspecting intermediate states.
    """
    def __init__(self, interval_steps=10, output_dir="checkpoints", filename_pattern="mycel_{step:04d}.json"):
        """
        Initialise the checkpoint saver.
        Args:
            interval_steps (int): Save a checkpoint every N steps.
            output_dir (str): Directory where checkpoint files will be stored.
            filename_pattern (str): Pattern for filenames, with '{step}' placeholder
        """
        # No. steps between saves
        self.interval_steps = interval_steps
        # Remember the last step saved to prevent duplicate saves
        self.last_step = -1
        # Create a Path for the output directory
        self.output_dir = Path(output_dir)
        # Ensure directory exists (creates parents as needed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Template for checkpoint filenames
        self.filename_pattern = filename_pattern

    def maybe_save(self, mycel, step):
        """
        Check if it's time to save a new checkpoint, and if so, write it.
        Args:
            mycel: Mycel simulation instance to serialise.
            step (int): Current simulation step number
        """
        # Only save when 'step' is a multiple of interval_steps and not already saved
        if step % self.interval_steps == 0 and step != self.last_step:
            # Construct the full filepath by formatting the pattern with the step
            filename = self.output_dir / self.filename_pattern.format(step=step)
            # Serialise the current state of 'mycel' to JSON at this location
            save_to_json(mycel, str(filename))
            # Update last_step to avoid saving again for this same step
            self.last_step = step
            logger.debug(f"Checkpoint saved @ step {step}: {filename}")
