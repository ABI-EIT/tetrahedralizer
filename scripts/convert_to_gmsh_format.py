import pyvista as pv
import pymeshfix
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pathlib
import os
import meshio
import numpy as np

"""
Convert to gmsh .msh file format
"""

output_directory = "output"
def main():
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh convert")
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
    m.write(f"{output_directory}/{path.stem}.msh", file_format="gmsh22")


if __name__ == "__main__":
    main()
