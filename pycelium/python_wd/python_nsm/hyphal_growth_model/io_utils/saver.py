# io_utils/saver.py

# Imports
import json # JSON serialisation for saving/loading state
from core.mycel import Mycel # Representing entire simulation
from core.point import MPoint # Reconstructing point/vector data
from core.section import Section # Section segments restored from JSON
from core.options import Options # Sim params

def save_to_json(mycel: Mycel, filename: str):
    """Save the current simulation state to a JSON file."""
    # Build a dictionary capturing the simulation state
    data = {
        # Simulation time (float)
        "time": mycel.time,
        # Options: covert the Options dataclass into a plain dict
        "options": vars(mycel.options),
        # List of section attributes to save
        "sections": [
            {
                # Starting coordinates as [x, y, z]
                "start": section.start.to_list(),
                # Ending coordinates as [x, y, z]
                "end": section.end.to_list(),
                # Orientation vector as [x, y, z]
                "orientation": section.orientation.to_list(),
                # Physical length of the segment
                "length": section.length,
                # Age of the segment
                "age": section.age,
                # Boolean flag: is this segment an active tip?
                "is_tip": section.is_tip,
                # Boolean flag: has this segment died?
                "is_dead": section.is_dead,
                # Index of this segment's parent in mycel.section, or None for the seed
                "parent_index": mycel.sections.index(section.parent) if section.parent else None
            }
            for section in mycel.sections # Iterate over all sections
        ]
    }
    # Write out the JSON file with indentation for readability
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved simulation to {filename}")

def load_from_json(filename: str) -> Mycel:
    """Load a saved simulation from JSON into a Mycel object."""
    # Read JSON data from disk
    with open(filename, "r") as f:
        data = json.load(f)

    # Reconstruct the Options object from the saved dict
    options = Options(**data["options"])
    # Create a new Mycel instance with these options
    mycel = Mycel(options)
    # Restore the simulation time
    mycel.time = data["time"]

    # Temporarily list to hold newly created Section instances
    sections = []
    for sec_data in data["sections"]:
        # Recreate MPoint instances for geometry and orientation
        start = MPoint(*sec_data["start"])
        end = MPoint(*sec_data["end"])
        orientation = MPoint(*sec_data["orientation"])

        # Instantiate a Section (opts and parent will be set later)
        sec = Section(start=start, orientation=orientation)
        # Restore geometric and state attributes
        sec.end = end
        sec.length = sec_data["length"]
        sec.age = sec_data["age"]
        sec.is_tip = sec_data["is_tip"]
        sec.is_dead = sec_data["is_dead"]
        sections.append(sec)

    # Now assign parent/children links
    for i, sec_data in enumerate(data["sections"]):
        parent_idx = sec_data["parent_index"]
        if parent_idx is not None:
            parent = sections[parent_idx]
            # Link this section to its parent
            sections[i].parent = parent
            # Add this section to the parent's children list
            parent.children.append(sections[i])

    # Assign the reconstructed sections list back to the Mycel object
    mycel.sections = sections
    
    print(f"✅ Loaded simulation from {filename}")
    return mycel
