# vis/animate_growth.py

# Imports
import pandas as pd # reading CSV time-series data
import matplotlib.pyplot as plt # Plotting
from mpl_toolkits.mplot3d import Axes3D # 3D plotting via mpl_toolkits
from matplotlib.animation import FuncAnimation # Creating animations
import os # Path manipulations
import logging
logger = logging.getLogger("pycelium")

def animate_growth(csv_path="outputs/mycelium_time_series.csv", save_path="outputs/mycelium_growth.mp4", interval=100):
    """
    Read a CSV of tip positions over time and create a 3D growth animation.
    Args:
        csv_path (str): path to CSV file containing columns [time, x, y, z]
        save_csv (str): path to save resulting MP4 (or fallback GIF)
        interval (int): delay between frames in ms
    """
    df = pd.read_csv(csv_path) # Load time-series data into a df
    steps = sorted(df["time"].unique()) # Extract distinct time steps and sort
    
    fig = plt.figure(figsize=(8, 6)) # Create 3D figure and axis for plotting
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Mycelium Growth Over Time") # Set initial title and axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatter = ax.scatter([], [], [], c='green', s=8) # Initialise scatter plot (empty for blitting 

    def update(frame_idx):
        """
        Update function for each animation frame.
        Args:
            frame_idx (int): Index into the sorted time steps list.
        """
        ax.cla() # Clear existing points and labels
        current_time = steps[frame_idx] # determine current sim time for this frame
        snapshot = df[df["time"] <= current_time] # select all points up to and including current time

        ax.set_title(f"Mycelium Growth @ t={current_time:.2f}") # Update title w/ current time
        ax.set_xlabel("X") # ensure axis labels and grid remain visible
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)

        ax.scatter(snapshot["x"], snapshot["y"], snapshot["z"], c='green', s=8) # Plot all tip positions recorded so far in green dots

        ax.set_xlim(df["x"].min(), df["x"].max()) # Fix axis limits to full data range for consistency
        ax.set_ylim(df["y"].min(), df["y"].max())
        ax.set_zlim(df["z"].min(), df["z"].max())

    ani = FuncAnimation(fig, update, frames=len(steps), interval=interval) # create animation object: calls update() for each frame

    try:
        ani.save(save_path, writer="ffmpeg", dpi=150) # Save as MP4 using ffmpeg
        logger.info(f"Animation saved: {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save MP4 with ffmpeg; falling back to GIF. Error: {e}") # on failure (e.g. ffmpeg not installed, fallback to GIF)
        fallback = save_path.replace(".mp4", ".gif")
        try:
            ani.save(fallback, writer="pillow", dpi=100)
            logger.info(f" Fallback GIF saved to {fallback}")
        except Exception as e2:
            logger.error(f"Failed to save fallback GIF: {e2}")
    plt.close() # close figure to release memory

if __name__ == "__main__":
    animate_growth() # If run as a script, execute animate_growth() with defaults
