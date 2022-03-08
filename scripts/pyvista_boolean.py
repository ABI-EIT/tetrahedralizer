import pyvista as pv
import numpy as np
import os
import pathlib
import meshio
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from matplotlib import cm
from pyvista_tools.pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d
import pymeshfix

output_directory = "output"
output_suffix = "unioned"


def main():
    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to union")
    if len(filenames) == 0:
        return
    Tk().destroy()
    meshes = [pv.PolyData(pv.read(filename)) for filename in filenames]

    # Plot input meshes
    cmap = cm.get_cmap("Set1")
    p = pv.Plotter()
    for i, mesh in enumerate(meshes):
        p.add_mesh(mesh, style="wireframe", opacity=0.5, color=cmap(i)[:-1], line_width=2)
    p.camera_position = [0, -1, 0.25]
    p.camera.focal_point = np.add(p.camera.focal_point, [0, 0, 25])
    p.add_title("Input Meshes")
    p.show()



    # Union meshes
    for mesh in meshes:
        mesh = mesh.clean()
        meshfix = pymeshfix.MeshFix(mesh.points, pyvista_faces_to_2d(mesh.faces))
        meshfix.repair()
        mesh.points = meshfix.v
        mesh.faces = pyvista_faces_to_1d(meshfix.f)

    booleaned = meshes[0].boolean_difference(meshes[1])

    meshes[0].plot_normals(mag=20)
    meshes[1].plot_normals(mag=20)

    booleaned.plot()


    # # Save result
    # output_filename = ""
    # for filename in filenames:
    #     mesh_path = pathlib.Path(filename)
    #     output_filename += f"{mesh_path.stem}_"
    # output_filename += output_suffix
    #
    # if not os.path.exists(output_directory):
    #     os.mkdir(output_directory)
    #
    # m = meshio.Mesh(combined.points, {"triangle": pyvista_faces_to_2d(combined.cells)})
    # m.write(f"{output_directory}/{output_filename}.stl")
    #
    # # Plot result
    # p = pv.Plotter()
    # p.add_mesh(combined, style="wireframe")
    # p.camera_position = [0, -1, 0.25]
    # p.camera.focal_point = np.add(p.camera.focal_point, [0, 0, 25])
    # p.add_title("Shared Faces Removed")
    # p.show()

if __name__ == "__main__":
    main()
