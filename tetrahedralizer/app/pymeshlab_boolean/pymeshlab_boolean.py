import pyvista as pv
import os
import pathlib
import meshio
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from tetrahedralizer.mesh_lib import pymeshlab_boolean
from pyvista_tools import pyvista_faces_to_1d, pyvista_faces_to_2d
import json


def main():
    # Load Config
    config_filename = "conf.json"
    with open(config_filename, "r") as f:
        config = json.load(f)

    output_directory = config["output_directory"]
    operation = config["operation"]
    output_suffix = operation
    output_extension = config["output_extension"]

    # Select files
    Tk().withdraw()
    filename_a = askopenfilename(title="Select mesh one")
    if filename_a == "":
        return
    Tk().destroy()

    Tk().withdraw()
    filename_b = askopenfilename(title="Select mesh two")
    if filename_b == "":
        return
    Tk().destroy()

    # Load files
    meshes = [pv.read(filename) for filename in [filename_a, filename_b]]

    # Run boolean operation
    booleaned_mesh = pymeshlab_boolean(meshes, operation)

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_filename = ""
    for filename in [filename_a, filename_b]:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    m = meshio.Mesh(booleaned_mesh.points, {"triangle": pyvista_faces_to_2d(booleaned_mesh.faces)})
    m.write(f"{output_directory}/{output_filename}{output_extension}")

    # Plot results
    blocks = pv.MultiBlock(meshes)
    combined = blocks.combine()
    p = pv.Plotter()
    p.add_mesh(combined, label="Input", style="wireframe")
    p.add_mesh(booleaned_mesh, color="red", style="wireframe", label="Result")
    p.add_title(f"Input Meshes with {operation} Result")
    p.add_legend(loc="lower right")
    p.show()


if __name__ == "__main__":
    main()
