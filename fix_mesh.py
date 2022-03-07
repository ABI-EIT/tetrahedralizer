import pyvista as pv
import pymeshfix
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pathlib
import os
import meshio
import numpy as np

"""
App to apply meshfix to mesh and save
"""

output_directory = "output"
output_suffix = "fixed"
plot_result = True

def main():
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to fix")
    if filename == "":
        return
    Tk().destroy()

    mesh = pv.read(filename)

    meshfix = pymeshfix.MeshFix(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    holes = meshfix.extract_holes()
    meshfix.repair()

    mesh.points = meshfix.v
    mesh.faces = np.insert(meshfix.f, 0, values=3, axis=1).ravel()

    path = pathlib.Path(filename)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    m = meshio.Mesh(mesh.points, {"triangle": mesh.faces.reshape(-1, 4)[:, 1:]})
    m.write(f"{output_directory}/{path.stem}_{output_suffix}.stl")

    if plot_result:
        p = pv.Plotter()
        p.add_mesh(mesh.extract_all_edges())
        if holes.number_of_points != 0:
            p.add_mesh(holes, color="r")
        p.show()


if __name__ == "__main__":
    main()
