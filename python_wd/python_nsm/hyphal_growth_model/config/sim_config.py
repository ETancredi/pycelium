# config/sim_config.py

# Imports
import json # JSON serialisation / deserialisation
from core.options import Options # Sim params
from argparse import ArgumentParser # Parsing cmd-line args in CLI mode

def load_options_from_json(path: str) -> Options:
    """
    Load sim opts from JSON file.
    Args:
        path (str): Filesystem path to the JSON config file.
    Returns:
        Options: an Options instance populated w/ values from the file.
    """
    with open(path, "r") as f: # Open the specified JSON file for reading
        data = json.load(f) # Parse the JSOn into a Python dict
    return Options(**data) # Unpack the dict into the Options dataclass and return

def load_options_from_cli() -> Options:
    """
    Parse cmd-line args to override or supply sim params.
    Recognised flags:
        --config <path> : JSON config file path
        --steps <int> : Override no. sim steps
    Returns:
        tuple:
            Options: sim opts (either from JSON or defualts)
            int: No. steps to run (or None if not provided)
    """
    # Create an arg parser 
    parser = ArgumentParser(description="Run hyphal growth simulation with custom parameters.")
    # Optional arg to specify a JSON config file
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    # Optional arg to override no. steps 
    parser.add_argument("--steps", type=int, help="Override number of simulation steps")
    # Parse args from sys.argv
    args = parser.parse_args()

    # If a config file path was provided, load options from it
    if args.config:
        opts = load_options_from_json(args.config)
    else:
        opts = Options()  # Otherwise, use default Opts values

    return opts, args.steps # Return both Options instance and parsed steps value
