# vis/animate2d.py

# Imports
import matplotlib.pyplot as plt # Plotting interface
import matplotlib.animation as animation # Animation support
from core.mycel import Mycel # Mycel simulation engine class
from core.options import Options # Sim params
from tropisms.orientator import Orientator # Oientator applies tropism rules per tip
import numpy as np # Numerical utilities

def animate_growth(mycel: Mycel, orientator: Orientator, steps=100, interval=200):
    """
    Animates the simulation frame-by-frame using matplotlib (2D).
    Args:
        mycel (Mycel): The simulation instance to animate.
        orientator (Orientator): applies grow direction adjustments.
        steps (int): No. animation frames (sim steps).
        interval (int): Delay between frames in ms.
    """
    fig, ax = plt.subplots(figsize=(6, 6)) # Create a figure and single axes for plotting
    ax.set_title("Hyphal Growth Animation") # Set window title and axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal") # Ensure equal scaling on both axes
    ax.grid(True) # Add grid lines for reference

    lines = [] # List to store plot artists (required by FuncAnimation)

    def init():
        """Initialisation function for FuncAnimation."""
        return lines

    def update(frame):
        """
        Update function called for each animation frame.
        Args:
            frame (int): Current fram index (0 to steps-1).
        """
        ax.clear() # clear previous frame's artists
        ax.set_title(f"Time = {mycel.time:.1f}") # Update title to show current sim time
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.grid(True)

        for tip in mycel.get_tips(): # Compute and apply new orientation for each active tip
            new_orientation = orientator.compute(tip) # compute new growth direction based on tropisms, fields, etc
            tip.orientation = new_orientation # Update tip's orientation in-place

        mycel.step() # Advance the sum by one step (grow, branch, etc)

        for section in mycel.get_all_segments(): # Draw all segments and tips on the 2D plot
            x0, y0 = section.start.coords[:2] # extract start and end coords in XY plane
            x1, y1 = section.end.coords[:2]
            ax.plot([x0, x1], [y0, y1], color="green", linewidth=1.2) # Plot the segment line in green
            if section.is_tip and not section.is_dead: # If this section is an alive tip, plot a red dot at its end
                ax.plot(x1, y1, "ro", markersize=3)

        return lines # Return list of artists for blitting 

    # Create the animation: call update() for each frame
    ani = animation.FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=interval)
    plt.show() # Display the animation window
