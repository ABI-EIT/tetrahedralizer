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

config_filename = "conf.json"


op_map = {
    "Intersection": pymeshlab.MeshSet.generate_boolean_intersection.__name__,
    "Union": pymeshlab.MeshSet.generate_boolean_union.__name__,
    "Difference": pymeshlab.MeshSet.generate_boolean_difference.__name__
}


def main():
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

    with open(config_filename, "r") as f:
        config = json.load(f)

    output_directory = config["output_directory"]
    operation = config["operation"]
    output_suffix = operation

    ms = pymeshlab.MeshSet()

    for filename in [filename_a, filename_b]:
        ms.load_new_mesh(filename)

    func = getattr(ms, op_map[operation])
    func(first_mesh=0, second_mesh=1)

    meshes = [ms.mesh(i) for i in range(ms.number_meshes())]
    original_meshes = [pv.PolyData(mesh.vertex_matrix(), pyvista_faces_to_1d(mesh.face_matrix())) for mesh in meshes[0:2]]
    blocks = pv.MultiBlock(original_meshes)
    combined = blocks.combine()

    pv_mesh = pv.PolyData(ms.mesh(2).vertex_matrix(), pyvista_faces_to_1d(ms.mesh(2).face_matrix()))

    # Save result
    output_filename = ""
    for filename in [filename_a, filename_b]:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    m = meshio.Mesh(pv_mesh.points, {"triangle": pyvista_faces_to_2d(pv_mesh.faces)})
    m.write(f"{output_directory}/{output_filename}.stl")

    # Plot results
    p = pv.Plotter()
    # Show input meshes in white
    p.add_mesh(combined, label="Input", style="wireframe")

    # Show resulting mesh in red
    p.add_mesh(pv_mesh, color="red", style="wireframe", label="Result")

    p.add_title(f"Input Meshes with {operation} Result")
    p.add_legend(loc="lower right")
    p.show()


if __name__ == "__main__":
    main()
