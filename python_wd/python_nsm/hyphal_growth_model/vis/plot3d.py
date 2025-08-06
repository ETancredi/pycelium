# vis/plot3d.py

# Imports
import matplotlib.pyplot as plt # Plotting
import os # Directory creation
from mpl_toolkits.mplot3d import Axes3D # Enable 3D plotting
from core.mycel import Mycel # Access sim segments and subsegments

def plot_mycel_3d(mycel: Mycel, title="Hyphal Growth in 3D", save_path=None):
    """
    Render the mycelium network in 3D, colouring each segment by its lineage colour
    Args:
        mycel (Mycel): Sim instance continaing all segments
        title (str): Figure title
        save_path (str, optional): Path to save the image. If None, defaults to 'outputs/mycelium_3d.png'.
    """
    fig = plt.figure(figsize=(8, 6)) # Create new 3D figure with specific size
    ax = fig.add_subplot(111, projection='3d') # add 3D axes
    ax.set_title(title) # Set plot title

    ax.set_xlabel("X") # Label axes for clarity
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    all_x, all_y, all_z = [], [], [] # Lists to collect all corrdinates for setting axis limits later

    for section in mycel.get_all_segments(): # Iterate through every section in sim
        for start, end in section.get_subsegments(): # Draw each stored subsegment for detailed geometry
            x0, y0, z0 = start.coords # Unpack start and end coords
            x1, y1, z1 = end.coords
            # draw each branch segment in its lineage color
            ax.plot([x0, x1], [y0, y1], [z0, z1],
                    color=section.color,
                    linewidth=1.2)
            all_x.extend([x0, x1]) # Collect coords for axis autoscaling
            all_y.extend([y0, y1])
            all_z.extend([z0, z1])

        if section.is_tip and not section.is_dead: # If this section is active, draw a circle marker at its end
            x_tip, y_tip, z_tip = section.end.coords
            # tip marker in the same RGB as its parent segment
            ax.scatter(x_tip, y_tip, z_tip, 
                       color=section.color, s=10)

    if all_x and all_y and all_z: # If we have any point, set axis limits to encompass all data
        ax.set_xlim(min(all_x), max(all_x))
        ax.set_ylim(min(all_y), max(all_y))
        ax.set_zlim(min(all_z), max(all_z))

    plt.tight_layout() # Adjust layout to prevent overlap

    if not save_path: # If no save_path provided, create 'outputs' directory and set default filename
        os.makedirs("outputs", exist_ok=True)
        save_path = "outputs/mycelium_3d.png"

    plt.savefig(save_path) # Save figure to disk and close to free resources
    plt.close()
