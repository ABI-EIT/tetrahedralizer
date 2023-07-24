import os
import pathlib
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import json
import pyvista as pv

from tetrahedralizer.mesh_lib import gmsh_tetrahedralize
from pyvista_tools import pyvista_faces_to_2d, pyvista_tetrahedral_mesh_from_arrays


def main():
    config_filename = "conf.json"
    with open(config_filename, "r") as f:
        config = json.load(f)

    output_directory = config["output_directory"]
    output_suffix = config["output_suffix"]
    output_extension = config["output_extension"]
    gmsh_options = config["gmsh_options"]

    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to gmsh")
    if len(filenames) == 0:
        return
    Tk().destroy()

    # Load files
    meshes = [pv.read(filename) for filename in filenames]

    # Tetrahedralize
    mesh = gmsh_tetrahedralize(meshes, gmsh_options)

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    mesh.save(f"{output_directory}/{output_filename}{output_extension}")

    #Plot result
    p = pv.Plotter()
    p.add_mesh(mesh.extract_all_edges(), opacity=0.15, show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = mesh.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", show_edges=True)

    p.add_plane_widget(plane_func, assign_to_axis="z")
    p.add_title("Tetrahedralized Mesh")
    p.show()


if __name__ == "__main__":
    main()
