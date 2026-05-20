# analysis/post_analysis.py

# Imports
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting figures and histograms
import logging 
logger = logging.getLogger("pycelium.analysis")
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from core.mycel import Mycel # Import Mycel class (main mycelium structure)
from core.section import Section # Import Section class (segment of mycelium)
from core.point import MPoint # Import MPoint class (seeding)

# Function to analyse branching angles in the mycelial network
def analyse_branching_angles(mycel: Mycel, save_path=None, csv_path=None):

    angles = [] # List to store computed angles between parent-child segment orientation

    # Loops over all sections (segments) of the mycelium
    for section in mycel.get_all_segments():
        for child in section.children: # Check each child of the current section
            if section.orientation and child.orientation: # Ensure both have valid orientation vectors
                dot = section.orientation.dot(child.orientation) # Compute dot product
                dot = max(min(dot, 1.0), -1.0) # Clamp value to valid arccos input range [-1, 1]
                angle_rad = np.arccos(dot) # Get angle in radians
                angles.append(np.degrees(angle_rad)) # Convert to degrees and store

    if not angles: # Exit if no valid angles found
        print("⚠️ No branching angles to analyse.")
        return

    mean_angle = np.mean(angles) # Calculate average branching angle
    logger.debug("Mean branching angle: {mean_angle:.2f}°", np.degrees(np.mean(angles)))

    # Plot and save histogram of branching angles if save_path is provided (if not, uses default)
    if save_path:
        plt.figure()
        plt.hist(angles, bins=20, color='royalblue')
        plt.title("Branching Angle Distribution")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.debug("Branching angle histogram saved to %s", save_path)
    
    # Export angles to CSV if csv_path is provided
    if csv_path:
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["BranchingAngleDegrees"])
            for angle in angles:
                writer.writerow([angle])
        logger.debug("Branching angles exported to %s", csv_path)

# Utility function to calculate angle in degrees between two vectors
def vector_angle_deg(v1, v2):
    """Returns angle in degrees between two vectors."""
    v1_u = v1 / np.linalg.norm(v1) # Normalise vector 1
    v2_u = v2 / np.linalg.norm(v2) # Normalise vector 2
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0) # Dot product, clamped to safe range
    return np.degrees(np.arccos(dot)) # Return angle in degrees

# Function to analuse and optionally export tip orientation vectors
def analyse_tip_orientations(mycel: Mycel, save_path=None, csv_path=None):

    tips = mycel.get_tips() # Retrieve tip segments of the mycelium
    if not tips: # if no tips are found, print warning and return
        print("⚠️ No tips to visualise orientations.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d") # Create 3D plot
    ax.set_title("Tip Orientation Vectors")

    orientations = [] # Store orientation vectors for CSV export

    # Loop through all tip segments and plot their orientation vectors
    for tip in tips:
        x, y, z = 0, 0, 0 # ALl vectors originate from origin for visualisation
        u, v, w = tip.orientation.coords # Get 3D orientation vector components
        orientations.append((u, v, w)) # Store for optional CSV export
        ax.quiver(x, y, z, u, v, w, length=1.0, normalize=True, color="purple") # Draw vector

    # Ser 3D plot limits and axis labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Save plot to file if path is provided (if not saves to deafult path)
    if save_path:
        plt.savefig(save_path)
        plt.close()
        logger.debug("Tip orientation histogram saved to %s", save_path)

    # Save orientation vectors to csv if path is provided
    if csv_path:
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["X", "Y", "Z"]) # Header
            for u, v, w in orientations:
                writer.writerow([u, v, w]) # Write each vector
        logger.debug("Orientation vectors exported to %s", csv_path)
