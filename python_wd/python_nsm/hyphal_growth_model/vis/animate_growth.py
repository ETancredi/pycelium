# vis/animate_growth.py

# Imports
import pandas as pd  # reading CSV time-series data
import matplotlib.pyplot as plt  # Plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting via mpl_toolkits
from matplotlib.animation import FuncAnimation  # Creating animations
import os  # Path manipulations
import logging

logger = logging.getLogger("pycelium")


def animate_growth(
    csv_path="outputs/mycelium_time_series.csv",
    save_path="outputs/mycelium_growth.mp4",
    interval=100,
):
    """
    Read a CSV of tip positions over time and create a 3D growth animation.
    Args:
        csv_path (str): path to CSV file containing columns [time, x, y, z]
        save_path (str): path to save resulting MP4 (or fallback GIF)
        interval (int): delay between frames in ms
    """
    df = pd.read_csv(csv_path)  # Load time-series data into a df

    if "time" not in df.columns or "x" not in df.columns or "y" not in df.columns or "z" not in df.columns:
        raise ValueError("CSV must contain 'time', 'x', 'y', 'z' columns for 3D animation.")

    steps = sorted(df["time"].unique())  # Extract distinct time steps and sort

    fig = plt.figure(figsize=(8, 6))  # Create 3D figure and axis for plotting
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Mycelium Growth Over Time")  # Set initial title and axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Initialise scatter plot (empty for blitting; we redraw each frame anyway)
    ax.scatter([], [], [], c="green", s=8)

    def update(frame_idx):
        """
        Update function for each animation frame.
        Args:
            frame_idx (int): Index into the sorted time steps list.
        """
        ax.cla()  # Clear existing points and labels
        current_time = steps[frame_idx]  # determine current sim time for this frame
        snapshot = df[df["time"] <= current_time]  # all points up to and including current time

        # Update title and labels
        ax.set_title(f"Mycelium Growth @ t={current_time:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)

        # Plot all tip positions recorded so far in green dots
        ax.scatter(snapshot["x"], snapshot["y"], snapshot["z"], c="green", s=8)

        # Fix axis limits to full data range for consistency
        ax.set_xlim(df["x"].min(), df["x"].max())
        ax.set_ylim(df["y"].min(), df["y"].max())
        ax.set_zlim(df["z"].min(), df["z"].max())

    ani = FuncAnimation(fig, update, frames=len(steps), interval=interval)

    try:
        ani.save(save_path, writer="ffmpeg", dpi=150)  # Save as MP4 using ffmpeg
        logger.info(f"Animation saved: {save_path}")
    except Exception as e:
        logger.warning(
            f"Failed to save MP4 with ffmpeg; falling back to GIF. Error: {e}"
        )
        fallback = save_path.replace(".mp4", ".gif")
        try:
            ani.save(fallback, writer="pillow", dpi=100)
            logger.info(f"Fallback GIF saved to {fallback}")
        except Exception as e2:
            logger.error(f"Failed to save fallback GIF: {e2}")
    plt.close()  # close figure to release memory


def animate_growth_2d(
    csv_path="outputs/mycelium_time_series.csv",
    save_path="outputs/mycelium_growth_2d.mp4",
    interval=100,
):
    """
    Read a CSV of tip positions over time and create a 2D growth animation (x–y only).

    Expects a CSV with at least columns:
        time, x, y
    (z is ignored if present).

    Args:
        csv_path (str): path to CSV file containing columns [time, x, y, (z)]
        save_path (str): path to save resulting MP4 (or fallback GIF)
        interval (int): delay between frames in ms
    """
    df = pd.read_csv(csv_path)

    if "time" not in df.columns or "x" not in df.columns or "y" not in df.columns:
        raise ValueError("CSV must contain 'time', 'x', 'y' columns for 2D animation.")

    # Distinct time steps
    steps = sorted(df["time"].unique())

    # Precompute axis limits for stable view
    xmin, xmax = df["x"].min(), df["x"].max()
    ymin, ymax = df["y"].min(), df["y"].max()

    # Add a little padding
    pad_x = 0.05 * max(1.0, xmax - xmin)
    pad_y = 0.05 * max(1.0, ymax - ymin)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Mycelium Growth Over Time (2D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    # Initial empty scatter
    scatter = ax.scatter([], [], c="green", s=8)

    def update(frame_idx):
        """
        Update function for each animation frame in 2D.
        """
        ax.cla()
        current_time = steps[frame_idx]
        snapshot = df[df["time"] <= current_time]

        ax.set_title(f"Mycelium Growth (2D) @ t={current_time:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        ax.scatter(snapshot["x"], snapshot["y"], c="green", s=8)

    ani = FuncAnimation(fig, update, frames=len(steps), interval=interval)

    try:
        ani.save(save_path, writer="ffmpeg", dpi=150)
        logger.info(f"2D animation saved: {save_path}")
    except Exception as e:
        logger.warning(
            f"Failed to save 2D MP4 with ffmpeg; falling back to GIF. Error: {e}"
        )
        fallback = save_path.replace(".mp4", ".gif")
        try:
            ani.save(fallback, writer="pillow", dpi=100)
            logger.info(f"Fallback 2D GIF saved to {fallback}")
        except Exception as e2:
            logger.error(f"Failed to save fallback 2D GIF: {e2}")
    plt.close()


if __name__ == "__main__":
    # Default behaviour if you run this file directly:
    # still call the original 3D animation
    animate_growth()
