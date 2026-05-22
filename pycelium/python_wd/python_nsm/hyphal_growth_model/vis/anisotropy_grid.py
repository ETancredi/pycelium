# vis/anisotropy_grid.py

# Imports
import numpy as np # Numerical ops
import matplotlib.pyplot as plt # Plotting library
from mpl_toolkits.mplot3d import Axes3D # Enables 3D aces in Matplotlib
from core.point import MPoint # 3D point/vector class for orientation vector

class AnisotropyGrid:
    """
    Spatial grid that stores a 3D bias (anisotropy vector) at each voxel.
    Useful for steering growth in preferred directions
    """
    def __init__(self, width=100, height=100, depth=100, resolution=10.0):
        """
        Initialise anisotropy field grid.
        Args:
            width (float): total size along x-axis
            height (float): total size along y-axis
            depth (float): total size along z-axis
            resolution (float): size of each grid cell.
        """
        # Save grid dimensions and resolution
        self.width = width
        self.height = height
        self.depth = depth
        self.resolution = resolution

        # Compute no. cells along each axis
        self.field = np.zeros((int(width / resolution),
                               int(height / resolution),
                               int(depth / resolution), 3))
        # Create a zero-initialised array of shape (nx, ny, nz, 3) for 3D vectors
        self.field[..., 0] = 1.0 # Default anisotropy: unit vector along +X axis for all cells

    def set_uniform_direction(self, direction: MPoint):
        """
        Set same anisotropy vector everywhere in the grid.
        Args:
            direction (MPoint): Desired bias direction (will be normalised).
        """
        vec = direction.normalise().as_array() # Normalise input direction and get its array form
        self.field[:, :, :, :] = vec # Broadcase this unit vector into every grid cell

    def get_direction_at(self, point: MPoint) -> MPoint:
        """
        Retrieve anisotropy direction at a given 3D point.
        Args:
            point (MPoint): query location in sim coords
        Returns:
            MPoint: normalised anisotropy vector at that location, or zero vector if outside the grid bounds.
        """
        # Convert world coords to grid indices (centered at origin)
        i = int((point.coords[0] + self.width / 2) / self.resolution)
        j = int((point.coords[1] + self.height / 2) / self.resolution)
        k = int((point.coords[2] + self.depth / 2) / self.resolution)

        if 0 <= i < self.field.shape[0] and 0 <= j < self.field.shape[1] and 0 <= k < self.field.shape[2]:
            vec = self.field[i, j, k] # Check if indices fall within grid dimensions and extract vector stored at [i,j,k]
            return MPoint(*vec).normalise() # Returns as a normalised MPoint
        else:
            return MPoint(0, 0, 0) # Outside grid: no bias

# Visualisation helper: 2D slice of anisotropy field in XY plane at Z=0
def plot_anisotropy_2d(grid: AnisotropyGrid, title="Anisotropy Vectors (XY Slice)", save_path=None):
    """
    Plot the anisotropy vectors on a 2D grid slide at the lowest Z layer.
    Args:
        grid (AnisotropyGrid: anisotropy grid object to visualise
        title (str): title for plot window
        save_path (str, optional): file path to save figure, if none, shows interactively
    """
    fig, ax = plt.subplots(figsize=(6, 6)) # Create 2D figure and axes
    ax.set_title(title) # Set plot title
    ax.set_xlabel("X") # label x-axis
    ax.set_ylabel("Y") # label y-axis

    step = 1 # Step size for sampling vectors to avoid overcrowdine
    for i in range(0, grid.field.shape[0], step): # Loop over grid cells in XY plane at k=0
        for j in range(0, grid.field.shape[1], step):
            vec = grid.field[i, j, 0] # Extract 2D vector at (i,j,0)
            x = (i * grid.resolution) - grid.width / 2 # Compute world coords of cell centre
            y = (j * grid.resolution) - grid.height / 2
            ax.quiver(
                x, y, # Arrow tail position
                vec[0], vec[1], # Arrow direction components
                angles='xy', # Interpret direction in XY-plane
                scale_units='xy', # Scale in data units
                scale=1.0, # No additional scaling
                color='blue', # Arrow colour
                width=0.002 # Arrow shaft width
            )

    ax.axis("equal") # Equal scaling on both axes
    ax.grid(True) # Show grid lines
    plt.tight_layout() # Adjust layout to fit

    if save_path:
        plt.savefig(save_path) # Save figure to disk
        plt.close()
    else:
        plt.show() # Display interactively

# Vis helper: sample and plot anisotropy in 3D
def plot_anisotropy_3d(grid: AnisotropyGrid, save_path=None):
    """
    Plot a subset of anisotropy vectors throughout 3D grid.
    Args:
        grid (AnisotropyGrid): the anisotropy grid object
        savepath (str, optional): File path to save figure, if none, shows interactively
    """
    # Create new 3D figure and axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Anisotropy Field (3D Sample)")

    step = 2 # Sampling stride for performance and clarity
    # Loop through sampled grid cells in 3D
    for i in range(0, grid.field.shape[0], step):
        for j in range(0, grid.field.shape[1], step):
            for k in range(0, grid.field.shape[2], step):
                vec = grid.field[i, j, k] # Extract 3D bias vector
                if np.linalg.norm(vec) < 1e-3: # Skip near-zero vectors to reduce plot clutter
                    continue
                # Convert indices back to world coordinates
                x = (i * grid.resolution) - grid.width / 2
                y = (j * grid.resolution) - grid.height / 2
                z = (k * grid.resolution) - grid.depth / 2
                # Draw 3D arrow at (x, y, z) pointing along vec
                ax.quiver(x, y, z, vec[0], vec[1], vec[2], length=3.0, normalize=True, color="blue") # same as 2D helper

    ax.set_xlabel("X") # Label axes
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout() # Adjust layout

    if save_path:
        plt.savefig(save_path) # Save figure
        plt.close()
    else:
        plt.show() # Display interactively
