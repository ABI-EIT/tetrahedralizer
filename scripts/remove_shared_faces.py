import pyvista as pv
import numpy as np
import os
import pathlib
import meshio
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from matplotlib import cm

from pyvista_tools.pyvista_tools import remove_shared_faces, pyvista_faces_to_2d, pyvista_faces_to_1d, \
    select_faces_using_points

"""
App to combine meshes by removing their shared walls. Designed to be used with surface meshes representing the lobes of
the lung. When tetrahedralizing a torso surface with lungs, this is necessary even though we wish to retain the surfaces
of the individual lobe. This is because our tetrahedralization approach is a two step process. First, tetrahedralize the
torso, leaving a hole for the lungs. Second, fill the hole with the lung lobes. In the first step therefore, we need a 
combined surface of the entire lung.
"""


def main():
    output_directory = "output"
    output_suffix = "shared_faces_removed"

    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to merge")
    if len(filenames) == 0:
        return
    Tk().destroy()

    meshes = [pv.PolyData(pv.read(filename)) for filename in filenames]

    # Combine lobes
    combined, removed_points = remove_shared_faces(meshes, return_removed_points=True)

    # Save result
    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    m = meshio.Mesh(combined.points, {"triangle": pyvista_faces_to_2d(combined.cells)})
    m.write(f"{output_directory}/{output_filename}.stl")

    # Plot input meshes
    cmap = cm.get_cmap("Set1")
    p = pv.Plotter()
    for i, mesh in enumerate(meshes):
        p.add_mesh(mesh, style="wireframe", opacity=0.5, color=cmap(i)[:-1], line_width=2)
    p.camera_position = [0, -1, 0.25]
    p.camera.focal_point = np.add(p.camera.focal_point, [0, 0, 25])
    p.add_title("Input Meshes")
    p.show()

    # Plot result
    shared_faces_meshes = []
    for mesh, points in zip(meshes, removed_points):
        shared_faces_indices = select_faces_using_points(mesh, points)
        shared_faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh.faces)[shared_faces_indices])
        shared_faces_meshes.append(pv.PolyData(mesh.points, faces=shared_faces))

    p = pv.Plotter()
    p.add_mesh(combined, style="wireframe")
    for mesh in shared_faces_meshes:
        p.add_mesh(mesh, style="wireframe", color="red")
    p.camera_position = [0, -1, 0.25]
    p.camera.focal_point = np.add(p.camera.focal_point, [0, 0, 25])
    p.add_title("Shared Faces Removed")
    p.show()


if __name__ == "__main__":
    main()


