import numpy as np
import struct
import matplotlib.pyplot as plt
import polyscope as ps
import trimesh
import typer
from typing import Optional

app = typer.Typer()


def read_voxel_grid(file_name: str):
    """Read a voxel grid from a file."""
    with open(file_name, "rb") as f:
        # Read the ASCII part of the header until the binary grid starts
        format_identifier = f.readline().decode("ascii").strip()  # e.g., G3
        data_type = f.readline().decode("ascii").strip()  # e.g., 1 FLOAT
        resolution_line = f.readline().decode("ascii").strip()  # e.g., "256 256 256"

        # Extract the resolution from the ASCII line
        res_x, res_y, res_z = map(int, resolution_line.split())

        # Skip the next few lines that describe the transformation matrix
        for _ in range(4):
            f.readline()

        # Now read the binary grid data (assuming float32 values)
        total_values = res_x * res_y * res_z
        grid_data = np.frombuffer(f.read(total_values * 4), dtype=np.float32)

    return grid_data.reshape((res_x, res_y, res_z)), (res_x, res_y, res_z)


def visualize_slices(grid: np.ndarray, res: tuple):
    """Visualize the voxel grid slices using matplotlib."""
    # Visualize a slice of the 3D grid (e.g., slice at the center along x)
    plt.imshow(grid[res[0] // 2], cmap="gray")
    plt.colorbar()
    plt.title("Slice of the Implicit Function (X plane)")
    plt.show()

    # Visualize a slice of the 3D grid (e.g., slice at the center along y)
    plt.imshow(grid[:, res[1] // 2], cmap="gray")
    plt.colorbar()
    plt.title("Slice of the Implicit Function (Y plane)")
    plt.show()

    # Visualize a slice of the 3D grid (e.g., slice at the center along z)
    plt.imshow(grid[:, :, res[2] // 2], cmap="gray")
    plt.colorbar()
    plt.title("Slice of the Implicit Function (Z plane)")
    plt.show()


def visualize_polyscope(grid: np.ndarray, res: tuple, mesh_file: Optional[str]):
    """Visualize the voxel grid and a mesh (if provided) together using Polyscope."""
    # Initialize Polyscope
    ps.init()

    # Define the grid dimensions and bounding box for visualization
    bound_low = (-1.0, -1.0, -1.0)  # Adjust to the lower bound of your grid space
    bound_high = (1.0, 1.0, 1.0)  # Adjust to the upper bound of your grid space

    # Register the volume grid with Polyscope
    ps_grid = ps.register_volume_grid(
        "Implicit Function Grid", res, bound_low, bound_high
    )

    # Add the scalar function on the grid (your implicit function values)
    ps_grid.add_scalar_quantity(
        "Implicit Function Values",
        grid,
        defined_on="nodes",
        vminmax=(grid.min(), grid.max()),
        enabled=True,
    )

    # If a mesh file is provided, load and visualize the mesh
    if mesh_file:
        mesh = trimesh.load(mesh_file)
        # Swap the X and Z axes if necessary (transposition fix)
        mesh.vertices[:, [0, 2]] = mesh.vertices[:, [2, 0]]
        # Register the mesh with Polyscope
        ps_mesh = ps.register_surface_mesh("Mesh", mesh.vertices, mesh.faces)

    # Show the visualization
    ps.show()


@app.command()
def main(
    grid_file: str,
    mesh_file: Optional[str] = None,
    vis_matplotlib: bool = True,
    vis_polyscope: bool = True,
):
    """
    Main function to visualize the voxel grid and mesh.

    - `grid_file`: Path to the voxel grid file.
    - `mesh_file`: Path to the mesh file (optional, only used with Polyscope).
    - `vis_matplotlib`: Visualize slices of the grid using Matplotlib.
    - `vis_polyscope`: Visualize the grid and mesh using Polyscope.
    """
    # Load the voxel grid
    grid, res = read_voxel_grid(grid_file)

    if vis_matplotlib:
        visualize_slices(grid, res)

    if vis_polyscope:
        visualize_polyscope(grid, res, mesh_file)


if __name__ == "__main__":
    app()
