import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pathlib
import os

from tetrahedralizer.mesh_lib import fix_mesh

"""
App to apply meshfix to mesh and save
"""


def main():
    output_directory = "output"
    output_suffix = "fixed"
    repair_kwargs = None
    output_extension = ".stl"

    # Select File
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to fix")
    if filename == "":
        return
    Tk().destroy()

    # Load File
    mesh = pv.read(filename)

    # Fix
    fixed_mesh, holes = fix_mesh(mesh, repair_kwargs)

    # Save
    path = pathlib.Path(filename)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    fixed_mesh.save(f"{output_directory}/{path.stem}_{output_suffix}{output_extension}")

    # Plot
    p = pv.Plotter()
    p.add_mesh(fixed_mesh.extract_all_edges())
    if holes.number_of_points != 0:
        p.add_mesh(holes, color="r")
    p.show()


if __name__ == "__main__":
    main()
