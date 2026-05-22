# analysis/stats_summary.py

# Imports
from core.mycel import Mycel # Mycelium data structure
from core.section import Section # Section (segment) of mycelium
import numpy as np # Numerical operators

# Function to print a summary of the final simulation statistics
def summarise(mycel: Mycel):
    segments = mycel.get_all_segments() # Retrieve all mycelial segments (active + dead)
    # Filter to get only tip segments that are still alive
    tips = [s for s in segments if s.is_tip and not s.is_dead]
    # Calculate depth for each tip (distance from seed)
    depths = [get_depth(s) for s in tips]
    # Collect segment ages, lengths, and no. children per segment
    ages = [s.age for s in segments]
    lengths = [s.length for s in segments]
    child_counts = [len(s.children) for s in segments]

    # Print formatted summary
    print("\n📊 FINAL SIMULATION SUMMARY")
    print(f"🧩 Total segments: {len(segments)}") # Total no. segments
    print(f"🌱 Active tips: {len(tips)}") # No. active tips
    print(f"📏 Avg segment length: {np.mean(lengths):.2f}") # Mean segment length
    print(f"⏳ Avg segment age: {np.mean(ages):.2f}") # Mean segment age
    print(f"🌿 Max depth from root: {max(depths) if depths else 0}") # Max distrance from seed
    print(f"🔗 Avg branches per node: {np.mean(child_counts):.2f}") # Average branching factor

# Helper function to compute depth for a given section
def get_depth(section: Section) -> int:
    """Recursively computes depth from root."""
    depth = 0 # Start at 0
    current = section
    while current.parent: # Traverse ipwards until the seed (no parent)
        current = current.parent
        depth += 1
    return depth # Return no. steps from seed
