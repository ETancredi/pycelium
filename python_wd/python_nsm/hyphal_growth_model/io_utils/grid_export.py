# io_utils/grid_export.py

# Imports
import numpy as np # Array saving and numerical operations
import matplotlib.pyplot as plt # Image export functionality
from vis.density_map import DensityGrid # Defining grid structure
import os # File path checks and creation

def export_grid_to_csv(grid: DensityGrid, filename: str):
    """
    Save the density grid data to a CSV file.
    Args:
        grid (DensityGrid): Object containing a 2D or 3D numpy array in 'grid.grid'.
        filename (str): Path where the CSV file will be written.
    """
    # Use NumPy's savetext to write the array to CSV with 4 decimal places
    np.savetxt(
        filename, # Output file path
        grid.grid, # The under-lying NumPy array to save
        delimiter=",", # Comma-separated values for CSV
        fmt="%.4f" # Format each number to 4 decimal places
    )
    # Inform the user that the CSV export succeeded
    print(f"✅ Density grid saved to CSV: {filename}")

def export_grid_to_png(grid: DensityGrid, filename: str, cmap="hot"):
    """
    Save the density grid as an image (heatmap) in PNG format.
    Args:
        grid (DensityGrid): Object containing the array in 'grid.grid'.
        filename (str): Path where the PNG file will be written.
        cmap (str): Matplotlib colormap name to map values to colours.
    """
    # Ensure output directory exists
    # Use Matplotlib's imsave to write the grid as an image
    plt.imsave(
        filename, # Output image file path
        grid.grid, # Data array to render
        cmap=cmap, # Colourmap for value-to-colour mapping
        origin='lower' # Place the [0,0] index at the lower-left of the image
    )
    # Inform the user that the PNG export succeeded 
    print(f"🖼️ Density heatmap saved as image: {filename}")
