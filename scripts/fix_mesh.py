import pyvista as pv
import pymeshfix
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pathlib
import os
import meshio
import numpy as np
from typing import Tuple, Dict, Union, Optional
from pyvista_tools import pyvista_faces_to_1d, pyvista_faces_to_2d

"""
App to apply meshfix to mesh and save
"""


def main():
    output_directory = "output"
    output_suffix = "fixed"
    repair_kwargs = None

    # Read
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to fix")
    if filename == "":
        return
    Tk().destroy()

    mesh = pv.read(filename)

    # Fix
    fixed_mesh, holes = fix_mesh(mesh, repair_kwargs)

    # Save
    path = pathlib.Path(filename)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    m = meshio.Mesh(fixed_mesh.points, {"triangle": pyvista_faces_to_2d(fixed_mesh.faces)})
    m.write(f"{output_directory}/{path.stem}_{output_suffix}.stl")

    # Plot
    p = pv.Plotter()
    p.add_mesh(fixed_mesh.extract_all_edges())
    if holes.number_of_points != 0:
        p.add_mesh(holes, color="r")
    p.show()


def fix_mesh(mesh: pv.DataSet, repair_kwargs: Dict = None) -> Tuple[pv.DataSet, pv.PolyData]:
    """
    Call the meshfix.repair function on a Pyvista dataset and return a fixed copy of the dataset along with the meshfix
    holes

    Parameters
    ----------
    mesh
        Pyvista Dataset
    repair_kwargs
        Kwargs for meshfix.repair

    Returns
    -------
        Fixed copy of the input mesh, holes identified by meshfix

    """
    if repair_kwargs is None:
        repair_kwargs = {}
    meshfix = pymeshfix.MeshFix(mesh.points, pyvista_faces_to_2d(mesh.faces))
    holes = meshfix.extract_holes()
    meshfix.repair(**repair_kwargs)
    fixed_mesh = mesh.copy()
    fixed_mesh.points = meshfix.v
    fixed_mesh.faces = pyvista_faces_to_1d(meshfix.f)

    return fixed_mesh, holes


if __name__ == "__main__":
    main()
