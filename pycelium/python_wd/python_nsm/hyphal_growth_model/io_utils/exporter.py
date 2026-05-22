# io_utils/exporter.py

# Imports
import os
import csv # Writing CSV files
import logging
logger = logging.getLogger("pycelium")
from core.mycel import Mycel # Type hinting and introspection of simulation state

def export_to_csv(mycel: Mycel, filename="mycelium.csv", all_time=False):
    """
    Export simulation data to a csv file.
    Args:
        mycel (Mycel): The simulation instance containing data to export.
        filename (str): Path to the output CSV file.
        all_time (bool): 
            - If True, write the tip time series over all steps.
            - If False, write the final network geometry and segment metadata.
    """
    # Open the file for writing, ensuring no extra blank lines 
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f) # Create a csv writer object

        if all_time:
            # Header for time-series export: step, infex, tip index, coords, age, length
            writer.writerow([
                "step", "tip_index",
                "x", "y", "z",
                "age", "length",
                "germ_tube_order", "germ_tube_role", "emergence_step", "emergence_time",
                "drug_mic", "drug_mic_parent",
                "drug_mic_mutated_from_parent", "drug_mic_mutated_from_wildtype",
                "drug_mic_mutation_delta_log", "drug_mic_mutation_probability",
                "mic_mutation_lineage_id", "mic_mutation_origin_section_id", "visual_color_source",
                "r", "g", "b",
                "drug_concentration",
                "drug_raw_growth_rate", "drug_effective_growth_rate", "drug_growth_multiplier",
                "drug_growth_stopped_by_drug", "drug_growth_stopped_time", "drug_growth_stop_reason",
                "drug_killed_by_drug", "drug_death_time"
            ])
            # Iterate over each recorded time step
            for step_idx, snapshot in enumerate(mycel.time_series):
                # Snapshot is a list of tip dicts for this step
                for i, tip in enumerate(snapshot):
                    # Extract fields in the correct order
                    row = [
                        step_idx, # simulation step no.
                        i, # index of tip in this snapshot
                        tip["x"], # x-coords of tip end
                        tip["y"], # y-coords
                        tip["z"], # z-coords
                        tip["age"], # age of tip segment
                        tip["length"], # length of tip segment
                        tip.get("germ_tube_order", ""), # germ-tube lineage index
                        tip.get("germ_tube_role", ""), # germ-tube role label
                        tip.get("emergence_step", ""), # scheduled/actual germ-tube emergence step
                        tip.get("emergence_time", ""), # simulation time at emergence
                        tip.get("drug_mic", ""), # MIC-like tolerance for this tip
                        tip.get("drug_mic_parent", ""), # Immediate-parent MIC before mutation
                        tip.get("drug_mic_mutated_from_parent", ""), # True if MIC changed on birth
                        tip.get("drug_mic_mutated_from_wildtype", ""), # True if MIC differs from wildtype
                        tip.get("drug_mic_mutation_delta_log", ""), # log(new_MIC / parent_MIC)
                        tip.get("drug_mic_mutation_probability", ""), # Effective mutation probability at birth
                        tip.get("mic_mutation_lineage_id", ""), # Colour lineage ID for this MIC breakout lineage
                        tip.get("mic_mutation_origin_section_id", ""), # Section ID where this colour lineage began
                        tip.get("visual_color_source", ""), # wildtype vs. MIC-mutant lineage colour source
                        tip.get("r", ""), # red colour channel
                        tip.get("g", ""), # green colour channel
                        tip.get("b", ""), # blue colour channel
                        tip.get("drug_concentration", ""), # sampled local drug concentration
                        tip.get("drug_raw_growth_rate", ""), # signed Ψ from the drug-response model
                        tip.get("drug_effective_growth_rate", ""), # non-negative rate actually passed to grow()
                        tip.get("drug_growth_multiplier", ""), # applied drug-derived growth multiplier
                        tip.get("drug_growth_stopped_by_drug", ""),
                        tip.get("drug_growth_stopped_time", ""),
                        tip.get("drug_growth_stop_reason", ""),
                        tip.get("drug_killed_by_drug", ""),
                        tip.get("drug_death_time", "")
                    ]
                    writer.writerow(row) # Write one row per tip per step
        else:
            # Header for final-segments export: segment metadata and geometry
            writer.writerow([
                "id", "parent_id",
                "x0", "y0", "z0", # start point coords
                "x1", "y1", "z1", # end point coords
                "length", "age", 
                "is_tip", "is_dead", 
                "r", "g", "b", # RGB colour channels
                "germ_tube_order", "germ_tube_role", "emergence_step", "emergence_time", # Germination lineage metadata
                "drug_mic", "drug_mic_parent", # Antifungal tolerance and inheritance diagnostics
                "drug_mic_mutated_from_parent", "drug_mic_mutated_from_wildtype",
                "drug_mic_mutation_delta_log", "drug_mic_mutation_probability",
                "mic_mutation_lineage_id", "mic_mutation_origin_section_id", "visual_color_source",
                "drug_concentration", # Antifungal response diagnostics
                "drug_raw_growth_rate", "drug_effective_growth_rate", "drug_growth_multiplier",
                "drug_growth_stopped_by_drug", "drug_growth_stopped_time", "drug_growth_stop_reason",
                "drug_killed_by_drug", "drug_death_time"
            ]) 
            # Iterate over every segment in the final network
            for s in mycel.get_all_segments():
                # Grab colour and parent id
                r, g, b = getattr(s, "color", (None, None, None))
                # Determine parent segment ID or empty if seed
                parent_id = s.parent.id if s.parent is not None else ""
                
                # Assemble the row with id and parent_id firt
                row = [
                    s.id,
                    parent_id,
                    *s.start.coords, # unpack x0, y0, z0
                    *s.end.coords, # unpack x1, y1, z1
                    s.length, # segment length
                    s.age, # segment age
                    s.is_tip, # boolean flag (active vs. inactive tip)
                    s.is_dead, # boolean flag (alive section vs. dead section) 
                    r, g, b, # colour channels
                    getattr(s, "germ_tube_order", ""), # germ-tube lineage index
                    getattr(s, "germ_tube_role", ""), # germ-tube role label
                    getattr(s, "emergence_step", ""), # emergence step for this germ-tube lineage
                    getattr(s, "emergence_time", ""), # emergence time for this germ-tube lineage
                    getattr(s, "drug_mic", ""), # MIC-like tolerance carried by the segment
                    getattr(s, "drug_mic_parent", ""), # Parent MIC before daughter mutation
                    getattr(s, "drug_mic_mutated_from_parent", ""), # Whether MIC changed relative to parent
                    getattr(s, "drug_mic_mutated_from_wildtype", ""), # Whether MIC differs from wildtype
                    getattr(s, "drug_mic_mutation_delta_log", ""), # log(new_MIC / parent_MIC)
                    getattr(s, "drug_mic_mutation_probability", ""), # Effective probability used at birth
                    getattr(s, "mic_mutation_lineage_id", ""), # Colour lineage ID for this MIC breakout lineage
                    getattr(s, "mic_mutation_origin_section_id", ""), # Section ID where this mutant colour lineage began
                    getattr(s, "visual_color_source", ""), # wildtype vs. MIC-mutant lineage colour source
                    getattr(s, "last_drug_concentration", ""), # Last local drug concentration sampled by the tip
                    getattr(s, "last_drug_raw_growth_rate", ""), # Signed Ψ from the drug-response model
                    getattr(s, "last_drug_effective_growth_rate", ""), # Non-negative growth rate actually applied
                    getattr(s, "last_drug_growth_multiplier", ""), # Applied growth multiplier after clamping
                    getattr(s, "drug_growth_stopped_by_drug", ""),
                    getattr(s, "drug_growth_stopped_time", ""),
                    getattr(s, "drug_growth_stop_reason", ""),
                    getattr(s, "drug_killed_by_drug", ""),
                    getattr(s, "drug_death_time", "")
                ]
                writer.writerow(row) # Write one row per segment
    # Inform user that exports completed
    logger.info(f"CSV exported: {filename}")

def export_to_obj(mycel: Mycel, filename="mycelium.obj"):
    """
    Export the mycelium network to a Wavefront .obj file.
    Segments are represented as coloured lines between vertices.
    Args:
        mycel (Mycel): The simulation instance.
        filename (str): Path to the ouput .obj file.
    """
    vertices = [] # List to collect vertex coords
    edges = [] # List of index paris defining line segments

    # Build vertices and edge indices for each segment
    for i, s in enumerate(mycel.get_all_segments()):
        # Extract start and end coords as arrays
        v_start = s.start.coords
        v_end = s.end.coords
        # Append both endpoints to the vertex list
        vertices.append(v_start)
        vertices.append(v_end)
        # Record an edge between the two most recently added vertices
        edges.append((2*i+1, 2*i+2))  

    # Open the OBJ file for writing
    with open(filename, "w") as f:
        # Write vertex definitions: one per line, with optional RGB colour channels
        for s in mycel.get_all_segments():
            x0, y0, z0 = s.start.coords # start point coordinates
            x1, y1, z1 = s.end.coords # end point coordinates
            # Default colour grey if segment has no colour mutation
            r, g, b = getattr(s, "color", (0.5, 0.5, 0.5))
            # Write two 'v' lines: one for each expoint, including RGB
            f.write(f"v {x0} {y0} {z0} {r} {g} {b}\n")
            f.write(f"v {x1} {y1} {z1} {r} {g} {b}\n")
        # Write line definitions: 'l v1 v2'
        for e in edges:
            f.write(f"l {e[0]} {e[1]}\n")            
    logger.info(f"OBJ exported: {filename}")

def export_tip_history(mycel, filename="mycelium_time_series.csv"):
    """
    Export tip positions and optional drug-response diagnostics to CSV.

    The animation code only requires time/x/y/z columns, but extra columns are
    useful for later analysis of which tips experienced high antifungal and how
    much their growth was inhibited.

    Args:
        mycel: The simulation instance.
        filename(str): Path to output CSV file.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: keep time/x/y/z first so existing animation readers still work.
        writer.writerow([
            "time", "x", "y", "z",
            "age", "length",
            "germ_tube_order", "germ_tube_role", "emergence_step", "emergence_time",
            "drug_mic", "drug_mic_parent",
            "drug_mic_mutated_from_parent", "drug_mic_mutated_from_wildtype",
            "drug_mic_mutation_delta_log", "drug_mic_mutation_probability",
            "mic_mutation_lineage_id", "mic_mutation_origin_section_id", "visual_color_source",
            "r", "g", "b",
            "drug_concentration",
            "drug_raw_growth_rate", "drug_effective_growth_rate", "drug_growth_multiplier",
            "drug_growth_stopped_by_drug", "drug_growth_stopped_time", "drug_growth_stop_reason",
            "drug_killed_by_drug", "drug_death_time"
        ])

        # Prefer the richer time_series snapshots created in Mycel.step().
        # Fall back to the older step_history structure if time_series is absent.
        if getattr(mycel, "time_series", None):
            for snapshot in mycel.time_series:
                for tip in snapshot:
                    writer.writerow([
                        f"{tip.get('time', 0.0):.2f}",
                        tip.get("x", ""),
                        tip.get("y", ""),
                        tip.get("z", ""),
                        tip.get("age", ""),
                        tip.get("length", ""),
                        tip.get("germ_tube_order", ""),
                        tip.get("germ_tube_role", ""),
                        tip.get("emergence_step", ""),
                        tip.get("emergence_time", ""),
                        tip.get("drug_mic", ""),
                        tip.get("drug_mic_parent", ""),
                        tip.get("drug_mic_mutated_from_parent", ""),
                        tip.get("drug_mic_mutated_from_wildtype", ""),
                        tip.get("drug_mic_mutation_delta_log", ""),
                        tip.get("drug_mic_mutation_probability", ""),
                        tip.get("mic_mutation_lineage_id", ""),
                        tip.get("mic_mutation_origin_section_id", ""),
                        tip.get("visual_color_source", ""),
                        tip.get("r", ""),
                        tip.get("g", ""),
                        tip.get("b", ""),
                        tip.get("drug_concentration", ""),
                        tip.get("drug_raw_growth_rate", ""),
                        tip.get("drug_effective_growth_rate", ""),
                        tip.get("drug_growth_multiplier", ""),
                        tip.get("drug_growth_stopped_by_drug", ""),
                        tip.get("drug_growth_stopped_time", ""),
                        tip.get("drug_growth_stop_reason", ""),
                        tip.get("drug_killed_by_drug", ""),
                        tip.get("drug_death_time", ""),
                    ])
        else:
            # Backward-compatible path for older Mycel objects.
            for time, tips in mycel.step_history:
                for x, y, z in tips:
                    writer.writerow([f"{time:.2f}", x, y, z, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""])
    logger.info(f"Tip history exported: {filename}")

def export_biomass_history(mycel: Mycel, filename: str):
    """
    Export the total biomass and tip coint at each time step to csv.
    Args:
        mycel (Mycel): Simulation instance.
        filename (str): Path to output CSV file.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: time, no. tips, total biomass
        writer.writerow(["time", "tips", "biomass"])
        # Iterate over recorded biomass values
        for i, biomass in enumerate(mycel.biomass_history):
            # Compute corresponding simulation time
            t = i * mycel.options.time_step
            # Determine tip count from step_history if available
            tips_count = len(mycel.step_history[i][1]) if i < len(mycel.step_history) else 0
            # Wite one row per time step
            writer.writerow([t, tips_count, biomass])
    logger.info(f"Biomass history exported: {filename}")
