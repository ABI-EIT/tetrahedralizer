import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import pathlib
import os
import meshio
import numpy as np
from matplotlib import cm

"""
App combine gmsh .msh files
"""

output_directory = "output"
output_suffix = "combined"
plot_result = True


def main():
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to combine")
    if len(filenames) == 0:
        return
    Tk().destroy()

    meshes = [pv.read(filename) for filename in filenames]
    for i, mesh in enumerate(meshes):
        mesh.clear_cell_data()
        mesh.cell_data["Scalar"] = np.asarray([i % len(meshes)] * mesh.n_cells)
        mesh.set_active_scalars("Scalar")

    p = pv.Plotter()
    cmap = cm.get_cmap("Accent")
    blocks = pv.MultiBlock(meshes)
    combined = blocks.combine()

    p.add_mesh(combined, opacity=0.15, cmap=cmap, scalars="Scalar", show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = combined.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", cmap=cmap, scalars="Scalar", show_edges=True)

    p.add_plane_widget(plane_func, assign_to_axis="z")

    p.add_title("Combined Tetrahedralized Lung Sections")
    p.show()

    #  TODO output to ply


if __name__ == "__main__":
    main()
