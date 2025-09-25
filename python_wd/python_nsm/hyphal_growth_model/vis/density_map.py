# vis/density_map.py

# Imports
import numpy as np  # Array ops and indexing
import matplotlib.pyplot as plt  # Plotting density heatmap
from core.point import MPoint  # MPoint for gradient vectors
from core.mycel import Mycel  # Extract segments when updating

# NEW: logger (quiet by default; controlled by PYCELIUM_LOG_LEVEL)
import logging
logger = logging.getLogger("pycelium")


class DensityGrid:
    """
    2D grid accumulating counts of hyphal segment endpoints.
    Each cell's value represents no. segments whose end falls within that cell.
    Supports querying raw density and computing gradients for avoidance behaviours.
    """
    def __init__(self, width=100, height=100, resolution=1.0):
        """
        Initialise density grid
        Args:
            width (float): total X span of grid in sim units
            heigh (float): total Y span of gird
            resolution (float): size of each grid cell
        """
        self.width = width  # Store overall dims and resolution
        self.height = height
        self.resolution = resolution
        # Compute no. rows and cols, create 2D array of zeros to accumulate counts
        self.grid = np.zeros((int(height / resolution), int(width / resolution)))

    def add_point(self, point: MPoint):
        """
        Increment grid cell corresponding to a point's XY location.
        Args:
            point (MPoint): 3D point; only X and Y are used
        """
        x, y = point.coords[0], point.coords[1]  # Extract X and Y coords
        i = int((y + self.height / 2) / self.resolution)  # Convert world coords to grid indices
        j = int((x + self.width / 2) / self.resolution)
        if 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]:  # Only add if indices are w/in grid bounds
            self.grid[i, j] += 1  # increment count

    def get_density_at(self, point: MPoint) -> float:
        """
        Return raw count at grid cell for a given point.
        Args:
            point (MPoint): 3D query point
        Returns:
            float: Count of segments in that cell, or 0 if out of bounds
        """
        x, y = point.coords[0], point.coords[1]
        i = int((y + self.height / 2) / self.resolution)
        j = int((x + self.width / 2) / self.resolution)
        if 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]:
            return self.grid[i, j]
        return 0.0

    def get_gradient_at(self, point: MPoint) -> MPoint:
        """
        Approximate density gradient at a point using central differences.
        Args:
            point (MPoint): location where gradient is evaluated.
        Returns:
            MPoint: Unit vector pointing toward increasing density.
                    Zero vector if at boundary or no variation.
        """
        x, y = point.coords[0], point.coords[1]
        i = int((y + self.height / 2) / self.resolution)
        j = int((x + self.width / 2) / self.resolution)

        # Only compute if point is not on outermost border
        if 1 <= i < self.grid.shape[0] - 1 and 1 <= j < self.grid.shape[1] - 1:
            # Central difference in X direction: f(x+dx) - f(x-dx) over 2*resolution
            d_dx = (self.grid[i, j + 1] - self.grid[i, j - 1]) / (2 * self.resolution)
            # Central difference in Y direction: f(y+dy) - f(y-dy)
            d_dy = (self.grid[i + 1, j] - self.grid[i - 1, j]) / (2 * self.resolution)
            return MPoint(d_dx, d_dy, 0).normalise()  # Create vector and normalise it to get direction

        return MPoint(0, 0, 0)  # At boundaries or if grid too small, return 0 gradient

    def update_from_mycel(self, mycel: Mycel):
        """
        Rebuild density grid by adding one point per segment end.
        This clears grid and then increments counts for each segment.
        """
        # Reset grid to zero before accumulating
        self.grid.fill(0)
        count = 0

        for section in mycel.get_all_segments():
            # Previously we printed each section's state; that was very chatty.
            # Now we simply accumulate and (optionally) log a compact debug note.
            self.add_point(section.end)
            count += 1

        # Debug-only summary (shown if PYCELIUM_LOG_LEVEL=DEBUG)
        logger.debug(f"DensityGrid updated with {count} contributing points.")

def plot_density(grid: DensityGrid, title="Hyphal Density Map", save_path=None):
    """
    Visualise density grid as a heatmap, auto-zooming to non-zero regions.
    Args:
        grid (DensityGrid): grid to plot
        title (str): plot title
        save_path (str, optional): If provided, path to save figure, otherwise, display interactively.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Find indices of non-zero cells to crop display
    nonzero_indices = np.argwhere(grid.grid > 0)
    if nonzero_indices.size > 0:
        i_min, j_min = nonzero_indices.min(axis=0)  # Determine bounding box of non-zero region
        i_max, j_max = nonzero_indices.max(axis=0)
        margin = 5  # add some padding around non-zero region

        # Apply margin and clamp to grid bounds
        i_min = max(i_min - margin, 0)
        i_max = min(i_max + margin, grid.grid.shape[0])
        j_min = max(j_min - margin, 0)
        j_max = min(j_max + margin, grid.grid.shape[1])

        grid_region = grid.grid[i_min:i_max, j_min:j_max]  # Extract cropped subregion for plotting

        extent = [  # Compute world coord extents for imshow
            (j_min * grid.resolution) - grid.width / 2,
            (j_max * grid.resolution) - grid.width / 2,
            (i_min * grid.resolution) - grid.height / 2,
            (i_max * grid.resolution) - grid.height / 2
        ]
    else:
        grid_region = grid.grid  # If grid is entirely zero, show full grid
        extent = [
            -grid.width / 2, grid.width / 2,
            -grid.height / 2, grid.height / 2
        ]

    # Add colour bar labelled "Density"
    cax = ax.imshow(grid_region, origin='lower', cmap='hot', extent=extent, aspect='equal')
    fig.colorbar(cax, ax=ax, label="Density")

    ax.set_title(title)  # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)
    plt.tight_layout()  # Turn off grid lines

    if save_path:
        plt.savefig(save_path)  # Save figure
        plt.close()
    else:
        plt.show()  # Display plot window
