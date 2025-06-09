# vis/plot2d.py

import matplotlib.pyplot as plt
import os
from core.mycel import Mycel

def plot_mycel(mycel: Mycel, title="Hyphal Network", save_path=None):
    """Plots all subsegments of a Mycel object in 2D (top-down X-Y view)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Plot all subsegments instead of just start → end
    for section in mycel.get_all_segments():
        # draw each subsegment in its lineage color
        for start, end in section.get_subsegments():
            x0, y0 = start.coords[:2]
            x1, y1 = end.coords[:2]
            ax.plot([x0, x1], [y0, y1], color=section.color, linewidth=1.5)

        # color the tip marker to match its segment (if alive)
        if section.is_tip and not section.is_dead:
            x_tip, y_tip = section.end.coords[:2]
            ax.plot(x_tip, y_tip, "o", color=section.color, markersize=3)
    
    ax.axis("equal")
    plt.grid(True)

    if not save_path:
        os.makedirs("outputs", exist_ok=True)
        save_path = "outputs/mycelium_2d.png"

    plt.savefig(save_path)
    plt.close()
