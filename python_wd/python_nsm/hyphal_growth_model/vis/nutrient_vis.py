# vis/nutrient_vis.py

# Imports
import matplotlib.pyplot as plt # 2D plotting
from mpl_toolkits.mplot3d import Axes3D # 3D axes support Matplotlib
from core.options import Options # Access nutrient source settings

def plot_nutrient_field_2d(opts: Options, ax=None, save_path=None):
    """
    Plot nutrient attractors and repellents on 2D X-Y plane.
    Args:
        opts (Options): Sim options containing nutrient_attractors and nutrient_repellents list.
        ax (matplotlib.axes.Axes, optional): Existing axes to draw on, if None, a new figure and axes are created.
        save_path (str, optional): File path to save figure; if None, plot shown interactively.
    """
    if ax is None: # If no axes provided, creat new figure and axes
        fig, ax = plt.subplots()

    ax.set_title("Nutrient Field (2D)") # Set title and axis labels for context
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot attractors
    for pos, strength in opts.nutrient_attractors: # Each entry in opts.nutrient_attractors is ((x,y,z), strength)
        x, y = pos[0], pos[1] # Unpack only X and Y components for 2D plotting
        # Plot green triangle marker at attractor location, label Attractor if strength > 0, or repellent if < 0
        ax.plot(x, y, "g^", label="Attractor" if strength > 0 else "Repellent") # g^ (green triangle marker)
        # Annotate marker w/ strength value, offset slighlty above
        ax.annotate(
            f"{strength:+.1f}", # format : +1.0 or -1.0
            (x, y), # Annotate at the data point
            textcoords="offset points",
            xytext=(0, 5), # 5 points above marker
            ha="centre" # horizontally centre the text
        )

    # Plot repellents
    for pos, strength in opts.nutrient_repellents:
        x, y = pos[0], pos[1]
        ax.plot(x, y, "rv") # red 'v' marker
        ax.annotate(f"{strength:+.1f}", (x, y), textcoords="offset points", xytext=(0, -10), ha="centre")

    ax.legend(["Attractor", "Repellent"]) # Add lefend for first occurences of each lable
    ax.grid(True) # Enable grid for easier spatial context
    ax.axis("equal") # Ensure equal scaling on both axes
    plt.tight_layout() # Tight layoud reduces whitespace

    if save_path: # Save or show depending on save_path argument
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_nutrient_field_3d(opts: Options, ax=None, save_path=None):
    """
    Plots nutrient sources in 3D (optional height/position awareness).
    Args:
        opts (Options): Sim options containing nutrient_attractors and nutrient_repellents list.
        ax (matplotlib.axes.Axes, optional): Existing axes to draw on, if None, a new figure and axes are created.
        save_path (str, optional): File path to save figure; if None, plot shown interactively.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.set_title("Nutrient Field (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for pos, strength in opts.nutrient_attractors:
        ax.scatter(*pos, color="green", marker="^", s=40)
        ax.text(*pos, f"{strength:+.1f}", color="green", size=8)

    for pos, strength in opts.nutrient_repellents:
        ax.scatter(*pos, color="red", marker="v", s=40)
        ax.text(*pos, f"{strength:+.1f}", color="red", size=8)

    ax.view_init(elev=20, azim=135) # Adjust viewing angle for better visibility (elevation=20°, azimuth=135°)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
