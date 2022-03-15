import pyvista as pv
import pymeshfix
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pathlib
import os
import meshio
import numpy as np
from matplotlib import cm
from pyvista_tools import pyvista_tetrahedral_mesh_from_arrays

"""
App to view a mesh using the pyvista plane widget
"""

def main():
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to view")
    if filename == "":
        return
    Tk().destroy()

    mesh = pv.read(filename)

    p = pv.Plotter()
    p.add_mesh(mesh, opacity=0.15, show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = mesh.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", show_edges=True)

    p.add_plane_widget(plane_func, assign_to_axis="z")
    p.show()


if __name__ == "__main__":
    main()
