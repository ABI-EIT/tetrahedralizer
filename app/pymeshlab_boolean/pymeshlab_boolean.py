import pymeshlab
import pyvista as pv
import numpy as np
import os
import pathlib
import meshio
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pyvista_tools.pyvista_tools import pyvista_faces_to_1d, pyvista_faces_to_2d
import json
from typing import Tuple, List

pymeshlab_op_map = {
    "Difference": pymeshlab.MeshSet.generate_boolean_difference.__name__,
    "Intersection": pymeshlab.MeshSet.generate_boolean_intersection.__name__,
    "Union": pymeshlab.MeshSet.generate_boolean_union.__name__,
    "Xor": pymeshlab.MeshSet.generate_boolean_xor.__name__
}


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
    mesh_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in meshes]

    # Run boolean operation
    booleaned_mesh = pymeshlab_boolean((mesh_arrays[0], mesh_arrays[1]), operation)
    pv_booleaned_mesh = pv.PolyData(booleaned_mesh[0], pyvista_faces_to_1d(booleaned_mesh[1]))

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_filename = ""
    for filename in [filename_a, filename_b]:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    m = meshio.Mesh(pv_booleaned_mesh.points, {"triangle": pyvista_faces_to_2d(pv_booleaned_mesh.faces)})
    m.write(f"{output_directory}/{output_filename}{output_extension}")

    # Plot results
    blocks = pv.MultiBlock(meshes)
    combined = blocks.combine()
    p = pv.Plotter()
    p.add_mesh(combined, label="Input", style="wireframe")
    p.add_mesh(pv_booleaned_mesh, color="red", style="wireframe", label="Result")
    p.add_title(f"Input Meshes with {operation} Result")
    p.add_legend(loc="lower right")
    p.show()


def pymeshlab_boolean(meshes: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], operation: str) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a pymesh boolean operation on two input meshes. The meshes are described as a platform agnostic Tuple of ndarrays
    representing mesh vertices and faces. The format is as follows:
        Tuple element 0: 3xN ndarray of float representing XYZ points in 3D space
        Tuple element 1: 3xN ndarray of int representing triangular faces composed of indices the points

    Parameters
    ----------
    meshes
        Tuple of two input meshes in array format
    operation
        Pymeshlab boolean operation name. String names are mapped to pymeshlab functions in ppymeshlab_op_map

    Returns
    -------
    booleaned_mesh
        Result of the boolean operation in array format


    """
    ms = pymeshlab.MeshSet()
    for mesh in meshes:
        ml_mesh = pymeshlab.Mesh(mesh[0], mesh[1])
        ms.add_mesh(ml_mesh)

    func = getattr(ms, pymeshlab_op_map[operation])
    func(first_mesh=0, second_mesh=1)

    booleaned_mesh = (ms.mesh(2).vertex_matrix(), ms.mesh(2).face_matrix())
    
    return booleaned_mesh


if __name__ == "__main__":
    main()
