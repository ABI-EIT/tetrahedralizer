import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import vtkmodules.util.vtkConstants
from matplotlib import cm
import numpy as np
from tetrahedralizer.pyvista_tools import pyvista_faces_to_2d, extract_faces_with_edges, pyvista_faces_to_1d
import os
import pathlib
import meshio
import collections

"""
App to extract surfaces from a scaffold mesh (labelled hedahedral volumetric mesh)
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
        section_mesh = pv.UnstructuredGrid({vtkmodules.util.vtkConstants.VTK_HEXAHEDRON: section_cells}, mesh.points)
        surface = section_mesh.extract_surface()

        surface = surface.clean() #Merges close points, turning some quads into tris
        surface = pv.UnstructuredGrid(surface) # Convert to UG to we can extract quads
        surface = pv.PolyData(surface.points, pyvista_faces_to_1d(surface.cells_dict[9]))
        surface = surface.clean() # Clean again to remove points left from tris
        surface = surface.fill_holes(1000) # Fill holes left by tris
        surface = surface.triangulate()

        # boundary_edges = surface.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
        #                                                feature_edges=False, manifold_edges=False)
        #
        # non_manifold_edges = surface.extract_feature_edges(boundary_edges=False, non_manifold_edges=True,
        #                                                feature_edges=False, manifold_edges=False)
        #
        # # folded_edges = surface.extract_feature_edges(feature_angle=355, boundary_edges=False, non_manifold_edges=False,
        # #                                                feature_edges=True, manifold_edges=False)
        #
        # boundary_faces = extract_faces_with_edges(surface, boundary_edges)
        # two_sides = [item for item, count in collections.Counter(boundary_faces).items() if count > 1]
        # bf = np.array([0 if not np.any(np.isin(boundary_faces, i)) else 1 for i in range(surface.n_faces)])
        # double_bf = np.array([0 if not np.any(np.isin(two_sides, i)) else 1 for i in range(surface.n_faces)])
        # surface["boundary_faces"] = bf + double_bf
        #
        # # non_manifold_faces = extract_faces_with_edges(surface, non_manifold_edges)
        # # two_non_manifold_faces = [item for item, count in collections.Counter(non_manifold_faces).items() if count > 1]
        # # nmf = np.array([0 if not np.any(np.isin(non_manifold_faces, i)) else 1 for i in range(surface.n_faces)])
        # # dnmf = np.array([0 if not np.any(np.isin(two_non_manifold_faces, i)) else 1 for i in range(surface.n_faces)])
        # # surface["non_manifold_faces"] = nmf + dnmf
        #
        # # folded_faces = extract_faces_with_edges(surface, folded_edges)
        # # surface["folded_faces"] = np.array([0 if not np.any(np.isin(folded_faces, i)) else 1 for i in range(surface.n_faces)])
        #
        # p = pv.Plotter()
        # p.add_mesh(surface, show_edges=True, scalars="boundary_faces", cmap=cm.get_cmap("Set1_r"))
        # # p.add_mesh(boundary_edges, color="red", label="Boundary Edges", line_width=2)
        # p.add_mesh(non_manifold_edges, color="purple", label="Non-manifold Edges", line_width=2)
        # # p.add_mesh(folded_edges, color="blue", label="Folded Edges", line_width=2)
        # p.add_legend()
        # p.show()
        # cleaned = surface.remove_cells(two_sides)

        # shrunk = surface.scale([0.99,0.99,0.99], inplace=False)
        # surface_centroid = np.mean(surface.points, axis=0)
        # shrunk_centroid = np.mean(shrunk.points, axis=0)
        # shrunk = shrunk.translate(surface_centroid-shrunk_centroid, inplace=False)
        # surface.select_enclosed_points(shrunk, check_surface=False)
        #
        # p = pv.Plotter()
        # p.add_mesh(shrunk, opacity=0.3, show_edges=True)
        # p.add_mesh(surface, style="points")
        # p.show()

        section_surfaces[section] = surface




    # Plot input meshes
    cmap = cm.get_cmap("Set1")  # Choose a qualitative colormap to distinguish meshes
    p = pv.Plotter()
    for i, (section, surface) in enumerate(section_surfaces.items()):
        p.add_mesh(surface, style="wireframe", opacity=0.5, color=cmap(i)[:-1], line_width=2)
    p.add_title("Scaffold Surfaces")
    p.show()

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
