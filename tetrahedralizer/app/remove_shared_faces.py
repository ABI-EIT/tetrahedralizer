import pyvista as pv
import numpy as np
import os
import pathlib
import meshio
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from matplotlib import cm

from tetrahedralizer.pyvista_tools import remove_shared_faces, pyvista_faces_to_2d, pyvista_faces_to_1d, \
    select_faces_using_points, remove_shared_faces_with_ray_trace

"""
App to combine meshes by removing their shared walls. Designed to be used with surface meshes representing the lobes of
the lung. When tetrahedralizing a torso surface with lungs, this is necessary even though we wish to retain the surfaces
of the individual lobe. This is because our tetrahedralization approach is a two step process. First, tetrahedralize the
torso, leaving a hole for the lungs. Second, fill the hole with the lung lobes. In the first step therefore, we need a 
combined surface of the entire lung.
"""

ray_length = 0.1
incidence_angle_tolerance = 0.1  # Total range in radians


def main():
    output_directory = "output"
    output_suffix = "shared_faces_removed"
    output_extension = ".stl"

    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to merge")
    if len(filenames) == 0:
        return
    Tk().destroy()

    # Load files
    meshes = [pv.PolyData(pv.read(filename)) for filename in filenames]

    # Combine lobes
    trimmed_meshes, removed_faces = \
        remove_shared_faces_with_ray_trace(meshes, ray_length=ray_length,
                                           incidence_angle_tolerance=incidence_angle_tolerance,
                                           return_removed_faces=True)
    combined = pv.PolyData()
    for mesh in trimmed_meshes:
        combined = combined.merge(mesh)

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    m = meshio.Mesh(combined.points, {"triangle": pyvista_faces_to_2d(combined.faces)})
    m.write(f"{output_directory}/{output_filename}{output_extension}")

    # Plot input meshes
    cmap = cm.get_cmap("Set1")  # Choose a qualitative colormap to distinguish meshes
    p = pv.Plotter()
    for i, mesh in enumerate(meshes):
        p.add_mesh(mesh, style="wireframe", opacity=0.5, color=cmap(i)[:-1], line_width=2)
    p.camera_position = [0, -1, 0.25]
    p.camera.focal_point = np.add(p.camera.focal_point, [0, 0, 25])
    p.add_title("Input Meshes")
    p.show()

    # Create polydata of removed faces
    shared_faces_meshes = []
    for mesh, faces in zip(meshes, removed_faces):
        shared_faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh.faces)[faces])
        shared_faces_meshes.append(pv.PolyData(mesh.points, faces=shared_faces))

    # Plot removed faces
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
