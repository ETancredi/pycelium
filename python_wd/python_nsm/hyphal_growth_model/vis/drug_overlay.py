# vis/drug_overlay.py

"""
Combined antifungal-field and mycelium visualisation utilities.

The normal Pycelium outputs keep the antifungal field and the hyphal network as
separate figures. This module adds a third figure type: the final mycelium
overlaid on top of the final antifungal concentration field. That makes it much
easier to see whether tips stalled at the drug front, crossed into drug, or
remained in the drug-free interior.
"""

# Imports
import os  # Filesystem helper for creating output directories when needed

import matplotlib.patheffects as path_effects  # White outline around black hyphae for readability
import matplotlib.pyplot as plt  # Main plotting library
from matplotlib.colors import LinearSegmentedColormap  # Build white-to-dark-blue drug colour map
import numpy as np  # Robust finite-value checks for colour scaling

from core.mycel import Mycel  # Type hint for the simulated mycelial network


def antifungal_white_to_dark_blue_cmap():
    """
    Return the standard antifungal concentration colour map.

    The mapping is deliberately single-hue and intuitive:

        0 drug / no drug  -> white
        high drug         -> dark blue

    A custom map is used instead of relying on a named Matplotlib map so the
    appearance stays stable across environments.
    """
    return LinearSegmentedColormap.from_list(
        "antifungal_white_to_dark_blue",
        ["#ffffff", "#08306b"],
    )


def _safe_vmax(concentration, requested_vmax=None):
    """
    Choose a safe upper colour-limit for antifungal heatmaps.

    requested_vmax:
        If the user provides drug_plot_max_concentration, use it. This is useful
        for comparing runs on the same colour scale, e.g. always 0..8xMIC.

    fallback:
        Otherwise use the maximum value in the current field. A tiny positive
        floor prevents Matplotlib warnings for all-zero drug fields.
    """
    if requested_vmax is not None:
        requested_vmax = float(requested_vmax)
        if requested_vmax > 0.0:
            return requested_vmax

    finite_values = np.asarray(concentration)[np.isfinite(concentration)]
    if finite_values.size == 0:
        return 1.0

    return max(float(np.max(finite_values)), 1e-9)


def plot_mycelium_on_drug_field(
    mycel: Mycel,
    drug_field,
    save_path=None,
    title="Mycelium on antifungal field",
    array=None,
    vmax=None,
    fungus_linewidth=1.4,
    fungus_halo_width=3.4,
    show_colorbar=True,
):
    """
    Plot the final hyphal network over the antifungal concentration field.

    Args:
        mycel:
            Completed Pycelium simulation object.
        drug_field:
            DrugField2D instance containing the current/final concentration grid.
        save_path:
            Optional output path. If omitted, writes to
            outputs/mycelium_drug_overlay.png.
        title:
            Figure title.
        array:
            Optional concentration array to plot instead of drug_field.concentration.
            This is mainly useful for debugging or future time-series overlays.
        vmax:
            Optional maximum colour scale. Set this to 8.0 if you want all
            antifungal plots to use the same 0..8xMIC colour scale.
        fungus_linewidth:
            Width of the visible black hyphal line.
        fungus_halo_width:
            Width of the white outline drawn underneath each hyphal line.
        show_colorbar:
            If True, draw a colourbar labelled as antifungal concentration.
    """
    # Choose the requested array or fall back to the live/final drug field.
    concentration = drug_field.concentration if array is None else array

    # Set a default save location that mirrors the other Pycelium visual outputs.
    if save_path is None:
        os.makedirs("outputs", exist_ok=True)
        save_path = "outputs/mycelium_drug_overlay.png"

    # Create a square-ish figure. The axes are locked to equal scaling below so
    # one simulation unit in x matches one simulation unit in y.
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot the antifungal field as the background. Low/no drug is white; high
    # drug is dark blue. vmin is fixed at zero because negative drug is invalid.
    image = ax.imshow(
        concentration,
        origin="lower",
        extent=[
            drug_field.config.x_min,
            drug_field.config.x_max,
            drug_field.config.y_min,
            drug_field.config.y_max,
        ],
        aspect="equal",
        cmap=antifungal_white_to_dark_blue_cmap(),
        vmin=0.0,
        vmax=_safe_vmax(concentration, vmax),
        zorder=0,
    )

    # Draw every hyphal subsegment over the drug background. Each line is drawn
    # as black with a white halo, so it remains visible on both dark-blue high
    # drug and white no-drug regions.
    line_effects = [
        path_effects.Stroke(linewidth=fungus_halo_width, foreground="white"),
        path_effects.Normal(),
    ]

    for section in mycel.get_all_segments():
        for start, end in section.get_subsegments():
            x0, y0 = start.coords[:2]
            x1, y1 = end.coords[:2]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="black",
                linewidth=fungus_linewidth,
                solid_capstyle="round",
                path_effects=line_effects,
                zorder=3,
            )

        # Mark living tips with small black points, also outlined in white. This
        # helps distinguish stalled tips at the drug front from internal branches.
        if section.is_tip and not section.is_dead:
            x_tip, y_tip = section.end.coords[:2]
            ax.scatter(
                [x_tip],
                [y_tip],
                s=14,
                c="black",
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )

    # Use the drug-field limits rather than the colony limits, so the overlay
    # shows the fungus in the full environmental context.
    ax.set_xlim(drug_field.config.x_min, drug_field.config.x_max)
    ax.set_ylim(drug_field.config.y_min, drug_field.config.y_max)
    ax.set_aspect("equal", adjustable="box")

    # Standard labels. The title can be overridden by callers if needed.
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # A light grid is useful on a plain mycelium plot, but visually clutters the
    # heatmap overlay, so it is intentionally disabled here.
    ax.grid(False)

    if show_colorbar:
        fig.colorbar(image, ax=ax, label="Antifungal concentration")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
