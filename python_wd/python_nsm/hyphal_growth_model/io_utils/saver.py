# io_utils/saver.py

import json
from core.mycel import Mycel
from core.point import MPoint
from core.section import Section
from core.options import Options, ToggleableFloat, ToggleableInt

def save_to_json(mycel: Mycel, filename: str):
    """Save the current simulation state to a JSON file."""
    data = {
        "time": mycel.time,
        "options": vars(mycel.options),
        "sections": [
            {
                "start": section.start.to_list(),
                "end": section.end.to_list(),
                "orientation": section.orientation.to_list(),
                "length": section.length,
                "age": section.age,
                "is_tip": section.is_tip,
                "is_dead": section.is_dead,
                "parent_index": (
                    mycel.sections.index(section.parent)
                    if section.parent else None
                )
            }
            for section in mycel.sections
        ]
    }

    def _json_default(o):
        # serialize our toggleable types as simple dicts
        if isinstance(o, (ToggleableFloat, ToggleableInt)):
            return {"enabled": o.enabled, "value": o.value}
        # fallback to built-in behavior (will raise if truly unserializable)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)

    print(f"✅ Saved simulation to {filename}")

def load_from_json(filename: str) -> Mycel:
    """Load a saved simulation from JSON into a Mycel object."""
    with open(filename, "r") as f:
        data = json.load(f)

    options = Options(**data["options"])
    mycel = Mycel(options)
    mycel.time = data["time"]

    sections: list[Section] = []
    for sec_data in data["sections"]:
        start = MPoint(*sec_data["start"])
        end = MPoint(*sec_data["end"])
        orientation = MPoint(*sec_data["orientation"])

        sec = Section(start=start, orientation=orientation)
        sec.end = end
        sec.length = sec_data["length"]
        sec.age = sec_data["age"]
        sec.is_tip = sec_data["is_tip"]
        sec.is_dead = sec_data["is_dead"]
        sections.append(sec)

    # Reconnect parent/children links
    for idx, sec_data in enumerate(data["sections"]):
        parent_idx = sec_data["parent_index"]
        if parent_idx is not None:
            parent = sections[parent_idx]
            sections[idx].parent = parent
            parent.children.append(sections[idx])

    mycel.sections = sections
    print(f"✅ Loaded simulation from {filename}")
    return mycel
