import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import cm
import numpy as np
from tetrahedralizer.pyvista_tools import pyvista_faces_to_2d, extract_faces_with_edges, pyvista_faces_to_1d
import os
import pathlib
import meshio
from vtkmodules.util.vtkConstants import VTK_HEXAHEDRON, VTK_QUAD

"""
App to extract surfaces from an organ scaffold mesh (labelled hexahedral volumetric mesh). These are output in vtk 
format from the MAP Client (https://map-client.readthedocs.io/en/latest/index.html) and currently have some degenerate 
faces.
"""


def main():
    output_directory = "output"
    output_suffix = "surface"
    output_extension = ".stl"

    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to extract surfaces")
    if filename == "":
        return
    Tk().destroy()

    mesh = pv.read(filename)

    section_surfaces = {}
    for section in mesh.cell_data:
        section_cells = pyvista_faces_to_2d(mesh.cells)[mesh.cell_data[section] == 1]
        section_mesh = pv.UnstructuredGrid({VTK_HEXAHEDRON: section_cells}, mesh.points)
        surface = section_mesh.extract_surface()

        surface = surface.clean()  # Merges close points. Degenerate quads are turned into tris
        surface = pv.UnstructuredGrid(surface)  # Convert to UnstructuredGrid to we can extract quads
        surface = pv.PolyData(surface.points, pyvista_faces_to_1d(surface.cells_dict[VTK_QUAD]))
        surface = surface.clean()  # Clean again to remove points left from tris
        surface = surface.fill_holes(1000)  # Fill holes left by tris
        surface = surface.triangulate()
        section_surfaces[section] = surface

    # Plot input meshes
    cmap = cm.get_cmap("Set1")  # Choose a qualitative colormap to distinguish meshes
    p = pv.Plotter()
    for i, (section, surface) in enumerate(section_surfaces.items()):
        p.add_mesh(surface, style="wireframe", opacity=0.5, color=cmap(i)[:-1], line_width=2)
    p.add_title("Scaffold Surfaces")
    p.show()

    # n_rows = n_cols = np.ceil(np.sqrt(len(section_surfaces.items()))).astype(int)
    # iter = np.nditer(np.empty((n_rows, n_cols)), flags=["multi_index"]) # Create numpy iterator so we can traverse the rows and cols of the plotter
    # p = pv.Plotter(shape=(n_rows, n_cols))
    # for (name, mesh), _ in zip(section_surfaces.items(), iter):
    #     p.subplot(*iter.multi_index)
    #     p.add_mesh(mesh, style="wireframe")
    #     p.add_title(name)
    #
    # p.link_views()
    # p.show()

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    for section, surface in section_surfaces.items():
        mesh_path = pathlib.Path(section)
        output_filename = f"{mesh_path.stem}_"
        output_filename += output_suffix

        m = meshio.Mesh(surface.points, {"triangle": pyvista_faces_to_2d(surface.faces)})
        m.write(f"{output_directory}/{output_filename}{output_extension}")


if __name__ == "__main__":
    main()
