# vis/animate_growth.py

# Imports
import pandas as pd # reading CSV time-series data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

def animate_growth(csv_path="outputs/mycelium_time_series.csv", save_path="outputs/mycelium_growth.mp4", interval=100):
    df = pd.read_csv(csv_path)
    steps = sorted(df["time"].unique())
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Mycelium Growth Over Time")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatter = ax.scatter([], [], [], c='green', s=8)

    def update(frame_idx):
        ax.cla()
        current_time = steps[frame_idx]
        snapshot = df[df["time"] <= current_time]

        ax.set_title(f"Mycelium Growth @ t={current_time:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)

        ax.scatter(snapshot["x"], snapshot["y"], snapshot["z"], c='green', s=8)

        ax.set_xlim(df["x"].min(), df["x"].max())
        ax.set_ylim(df["y"].min(), df["y"].max())
        ax.set_zlim(df["z"].min(), df["z"].max())

    ani = FuncAnimation(fig, update, frames=len(steps), interval=interval)

    try:
        ani.save(save_path, writer="ffmpeg", dpi=150)
        print(f"🎥 Saved animation to {save_path}")
    except Exception as e:
        print(f"⚠️ Failed to save MP4: {e}")
        fallback = save_path.replace(".mp4", ".gif")
        ani.save(fallback, writer="pillow", dpi=100)
        print(f"🎞️ Fallback GIF saved to {fallback}")

    plt.close()

if __name__ == "__main__":
    animate_growth()
