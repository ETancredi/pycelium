# vis/plot2d.py

# Imports:
import matplotlib.pyplot as plt # Create 2D plots
import os # Handling directory creation
from core.mycel import Mycel # Imports Mycel sim class to access segments and subsegments

def plot_mycel(mycel: Mycel, title="Hyphal Network", save_path=None):
    """
    Plots all subsegments of a Mycel object in 2D (top-down X-Y view).
    Args:
        mycel (Mycel): Sim instance containing all sections
        title (str): Title of plot window
        save_path (str, optional): File path to save the figure. 
                                   If None, defaults to 'outputs/mycelium_2d.png'.
    """
    fig, ax = plt.subplots(figsize=(6, 6)) # Create new figure and axes for plotting
    ax.set_title(title) # Set title and axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot all subsegments instead of just start → end
    for section in mycel.get_all_segments():
        # draw each subsegment in its lineage color
        for start, end in section.get_subsegments():
            x0, y0 = start.coords[:2] # extract X and Y coords of subsegment endpoints
            x1, y1 = end.coords[:2]
            ax.plot( # Plot line segment in section's assigned colour
                [x0, x1], # X coords
                [y0, y1], # Y coords
                color=section.color, # Use segment colour for lineage visualisation
                linewidth=1.5) # Line thickness

        # color the tip marker to match its segment (if alive)
        if section.is_tip and not section.is_dead:
            x_tip, y_tip = section.end.coords[:2] # extract tip's end coords
            ax.plot(x_tip, y_tip, "o", color=section.color, markersize=3) # plot a small circle marker at tip in same colour
    
    ax.axis("equal") # Ensure equal scaling so network isn't distorted
    plt.grid(True) # Turn on grid for spatial reference

    if not save_path: # If no save path provided, create 'outputs' dir and set default filename
        os.makedirs("outputs", exist_ok=True)
        save_path = "outputs/mycelium_2d.png"

    plt.savefig(save_path) # Save fig. to disk and close to free memory
    plt.close()
