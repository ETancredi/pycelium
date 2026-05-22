# vis/animate_growth.py

# Imports
import pandas as pd  # reading CSV time-series data
import matplotlib.pyplot as plt  # Plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting via mpl_toolkits
from matplotlib.animation import FuncAnimation  # Creating animations
from matplotlib.collections import LineCollection  # Efficiently draw many hyphal segments per frame
from matplotlib.colors import LinearSegmentedColormap  # White-to-dark-blue antifungal heatmap
import numpy as np  # Array handling for drug fields and scatter offsets
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

        # Plot all tip positions recorded so far. If RGB columns are present,
        # use them so mutant lineages keep the same colours in the 3D MP4.
        if {"r", "g", "b"}.issubset(snapshot.columns):
            colours = snapshot[["r", "g", "b"]].to_numpy(dtype=float)
            ax.scatter(snapshot["x"], snapshot["y"], snapshot["z"], c=colours, s=8)
        else:
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

        if {"r", "g", "b"}.issubset(snapshot.columns):
            colours = snapshot[["r", "g", "b"]].to_numpy(dtype=float)
            ax.scatter(snapshot["x"], snapshot["y"], c=colours, s=8)
        else:
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



def _antifungal_white_to_dark_blue_cmap():
    """Return the same single-hue drug colour map used by final overlay PNGs."""
    return LinearSegmentedColormap.from_list(
        "antifungal_white_to_dark_blue",
        ["#ffffff", "#08306b"],
    )


def _safe_animation_vmax(drug_field_history=None, fallback_array=None, requested_vmax=None):
    """Choose a stable colour-scale maximum for the whole animation."""
    if requested_vmax is not None:
        requested_vmax = float(requested_vmax)
        if requested_vmax > 0.0:
            return requested_vmax

    max_value = 0.0
    if drug_field_history:
        for frame in drug_field_history:
            finite = np.asarray(frame)[np.isfinite(frame)]
            if finite.size:
                max_value = max(max_value, float(np.max(finite)))

    if fallback_array is not None:
        finite = np.asarray(fallback_array)[np.isfinite(fallback_array)]
        if finite.size:
            max_value = max(max_value, float(np.max(finite)))

    return max(max_value, 1e-9)


def _network_limits(network_history, drug_field=None):
    """Return stable x/y axis limits for a 2D network animation."""
    if drug_field is not None:
        return (
            float(drug_field.config.x_min),
            float(drug_field.config.x_max),
            float(drug_field.config.y_min),
            float(drug_field.config.y_max),
        )

    xs = []
    ys = []
    for frame in network_history:
        for (x0, y0), (x1, y1) in frame.get("segments", []):
            xs.extend([x0, x1])
            ys.extend([y0, y1])
        for x, y in frame.get("tips", []):
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return -1.0, 1.0, -1.0, 1.0

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = 0.05 * max(1.0, xmax - xmin)
    pad_y = 0.05 * max(1.0, ymax - ymin)
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def animate_network_growth_2d(
    network_history,
    save_path="outputs/mycelium_growth_2d.mp4",
    interval=100,
    drug_field=None,
    drug_field_history=None,
    vmax=None,
    dpi=150,
    show_colorbar=True,
    highlight_zero_growth=True,
    zero_growth_color=(1.0, 0.15, 0.0),
    zero_growth_linewidth=2.6,
    zero_growth_marker_size=34.0,
):
    """
    Create a 2D MP4 from full network snapshots instead of tip dots.

    Args:
        network_history:
            List of frame dictionaries made by Mycel._record_network_snapshot().
            Each frame contains "segments" as x/y line pairs, "tips" as active
            tip coordinates, and "time" as the simulation time.
        save_path:
            Destination MP4 path.  If ffmpeg is unavailable, a GIF fallback is
            written next to it.
        interval:
            Delay between frames in milliseconds.
        drug_field:
            Optional DrugField2D object.  When supplied, the animation uses the
            full antifungal-field extent and draws a concentration heatmap behind
            the mycelium.
        drug_field_history:
            Optional list of concentration arrays, one per simulation frame.
            This lets the movie show antifungal diffusion over time.
        vmax:
            Optional fixed colour-scale maximum for the drug heatmap.
        dpi:
            Resolution used when saving the MP4.
        show_colorbar:
            Whether to include an antifungal concentration colourbar.
    """
    if not network_history:
        raise ValueError("No network history was recorded; cannot create a line-based 2D MP4.")

    drug_field_history = drug_field_history or []
    has_drug = drug_field is not None

    xmin, xmax, ymin, ymax = _network_limits(network_history, drug_field=drug_field)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    image = None
    if has_drug:
        first_array = (
            drug_field_history[0]
            if len(drug_field_history) > 0
            else drug_field.concentration
        )
        image = ax.imshow(
            first_array,
            origin="lower",
            extent=[
                drug_field.config.x_min,
                drug_field.config.x_max,
                drug_field.config.y_min,
                drug_field.config.y_max,
            ],
            aspect="equal",
            cmap=_antifungal_white_to_dark_blue_cmap(),
            vmin=0.0,
            vmax=_safe_animation_vmax(
                drug_field_history=drug_field_history,
                fallback_array=drug_field.concentration,
                requested_vmax=vmax,
            ),
            zorder=0,
        )
        if show_colorbar:
            fig.colorbar(image, ax=ax, label="Antifungal concentration")

    # Draw a thick white collection underneath a thinner black collection.  This
    # reproduces the final overlay PNG style and keeps hyphae visible on both the
    # white drug-free centre and the dark-blue high-drug edge.
    halo_lines = LineCollection(
        [],
        colors="white",
        linewidths=3.4 if has_drug else 2.4,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    hypha_lines = LineCollection(
        [],
        colors=[],
        linewidths=1.4,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    ax.add_collection(halo_lines)
    ax.add_collection(hypha_lines)

    stopped_lines = LineCollection(
        [],
        colors=[zero_growth_color],
        linewidths=zero_growth_linewidth,
        capstyle="round",
        joinstyle="round",
        zorder=5,
    )
    ax.add_collection(stopped_lines)

    stopped_tip_scatter = ax.scatter(
        [],
        [],
        marker="x",
        s=zero_growth_marker_size,
        c=[zero_growth_color],
        linewidths=1.2,
        zorder=6,
    )

    tip_scatter = ax.scatter(
        [],
        [],
        s=14 if has_drug else 10,
        c=[],
        edgecolors="white" if has_drug else "none",
        linewidths=0.6 if has_drug else 0.0,
        zorder=4,
    )

    title = "Mycelium on antifungal field" if has_drug else "Mycelium growth"

    def update(frame_idx):
        frame = network_history[frame_idx]
        segments = frame.get("segments", [])
        segment_colors = frame.get("segment_colors", [])
        tips = frame.get("tips", [])
        tip_colors = frame.get("tip_colors", [])
        stopped_segments = frame.get("stopped_segments", []) if highlight_zero_growth else []
        stopped_tips = frame.get("stopped_tips", []) if highlight_zero_growth else []

        if image is not None and drug_field_history:
            drug_idx = min(frame_idx, len(drug_field_history) - 1)
            image.set_data(drug_field_history[drug_idx])

        halo_lines.set_segments(segments)
        hypha_lines.set_segments(segments)
        if segment_colors:
            hypha_lines.set_color(segment_colors)
        else:
            hypha_lines.set_color("black" if has_drug else "green")

        if tips:
            tip_scatter.set_offsets(np.asarray(tips, dtype=float))
            if tip_colors:
                tip_scatter.set_facecolor(np.asarray(tip_colors, dtype=float))
            else:
                tip_scatter.set_facecolor("black" if has_drug else "green")
        else:
            tip_scatter.set_offsets(np.empty((0, 2)))
            tip_scatter.set_facecolor(np.empty((0, 4)))

        stopped_lines.set_segments(stopped_segments)
        if stopped_tips:
            stopped_tip_scatter.set_offsets(np.asarray(stopped_tips, dtype=float))
        else:
            stopped_tip_scatter.set_offsets(np.empty((0, 2)))

        current_time = frame.get("time", frame_idx)
        ax.set_title(f"{title} @ t={current_time:.2f}")
        return halo_lines, hypha_lines, tip_scatter, stopped_lines, stopped_tip_scatter

    ani = FuncAnimation(fig, update, frames=len(network_history), interval=interval, blit=False)

    try:
        ani.save(save_path, writer="ffmpeg", dpi=dpi)
        logger.info("2D line-based animation saved: %s", save_path)
    except Exception as e:
        logger.warning(
            "Failed to save 2D line-based MP4 with ffmpeg; falling back to GIF. Error: %s",
            e,
        )
        fallback = save_path.replace(".mp4", ".gif")
        try:
            ani.save(fallback, writer="pillow", dpi=max(80, min(int(dpi), 150)))
            logger.info("Fallback 2D line-based GIF saved: %s", fallback)
        except Exception as e2:
            logger.error("Failed to save fallback line-based GIF: %s", e2)
    plt.close(fig)
