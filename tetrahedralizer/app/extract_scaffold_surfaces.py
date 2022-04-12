from __future__ import annotations
import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import cm
import numpy as np
from tetrahedralizer.pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d, \
    rewind_faces_to_normals, select_intersecting_triangles, \
    refine_surface, fill_holes
import os
import pathlib
import meshio
from vtkmodules.util.vtkConstants import VTK_HEXAHEDRON, VTK_QUAD, VTK_TRIANGLE

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
    # section_surfaces = approach_a(mesh)

    section_surfaces = {}
    for section in mesh.cell_data:
        section_cells = pyvista_faces_to_2d(mesh.cells)[mesh.cell_data[section] == 1]
        section_mesh = pv.UnstructuredGrid({VTK_HEXAHEDRON: section_cells}, mesh.points)
        surface = section_mesh.extract_surface()
        surface = surface.clean()  # Merges close points. Degenerate quads are turned into tris
        surface = surface.triangulate()
        surface = refine_surface(surface)
        surface = rewind_faces_to_normals(surface)
        section_surfaces[section] = surface

    manifold = [surface.is_manifold for surface in section_surfaces.values()]
    if np.all(manifold):
        print("All output surfaces manifold")
    else:
        non_manifold_indices = np.where(np.logical_not(manifold))[0].astype(str)
        print("Non manifold surfaces: " + ", ".join(non_manifold_indices))

    intersecting_faces = [select_intersecting_triangles(surface, justproper=True) for surface in section_surfaces.values()]
    non_intersecting = [not np.any(intersecting) for intersecting in intersecting_faces]
    if np.all(non_intersecting):
        print("All output surfaces non intersecting")
    else:
        non_intersecting_indices = np.where(np.logical_not(non_intersecting))[0].astype(str)
        print("Surfaces with intersecting faces: " + ", ".join(non_intersecting_indices))

    # Plot input meshes
    cmap = cm.get_cmap("Set1")  # Choose a qualitative colormap to distinguish meshes
    p = pv.Plotter()
    for i, (section, surface) in enumerate(section_surfaces.items()):
        p.add_mesh(surface, style="wireframe", opacity=0.5, color=cmap(i)[:-1], line_width=2)
    p.add_title("Scaffold Surfaces")
    p.show()

    # Plot all the surfaces in a square multi window plot
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


def approach_a(mesh):
    section_surfaces = {}
    for section in mesh.cell_data:
        section_cells = pyvista_faces_to_2d(mesh.cells)[mesh.cell_data[section] == 1]
        section_mesh = pv.UnstructuredGrid({VTK_HEXAHEDRON: section_cells}, mesh.points)
        surface = section_mesh.extract_surface()
        surface = surface.clean()  # Merges close points. Degenerate quads are turned into tris
        surface = pv.UnstructuredGrid(surface)  # Convert to UnstructuredGrid to we can extract quads
        surface = pv.PolyData(surface.points, pyvista_faces_to_1d(surface.cells_dict[VTK_QUAD]))
        surface = surface.clean()  # Clean again to remove points left from tris
        surface = surface.triangulate()
        # surface = repeatedly_fill_holes(surface, max_iterations=10, hole_size=1000)  # Fill holes left by tris
        surface = fill_holes(surface, strategy="nearest_neighbor")
        surface = rewind_faces_to_normals(surface)
        section_surfaces[section] = surface
    return section_surfaces


def plot_with_non_manifold(mesh: pv.PolyData, title=None):
    p = pv.Plotter()
    nm = mesh.extract_feature_edges(feature_edges=False, boundary_edges=False,
                                    manifold_edges=False, non_manifold_edges=True)

    p.add_mesh(mesh, color="white", style="wireframe", label="Surface Mesh")
    p.add_mesh(nm, color="red", label="Non-Manifold Edges")
    p.add_legend()
    if title is not None:
        p.add_title(title)
    p.show()


if __name__ == "__main__":
    main()
