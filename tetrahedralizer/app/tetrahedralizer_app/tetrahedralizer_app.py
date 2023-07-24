from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
import json
import pyvista as pv
from tetrahedralizer.mesh_lib import preprocess_and_tetrahedralize
from matplotlib import cm
import os
import pathlib


"""
App to automate the steps of creating a tetrahedralized mesh from one outer surface and several inner surfaces
"""


def main():
    # Load config
    config_filename = "conf.json"
    with open(config_filename, "r") as f:
        config = json.load(f)

    output_directory = config["output_directory"]
    output_suffix = config["output_suffix"]
    output_extension = config["output_extension"]
    mesh_repair_kwargs = config["mesh_repair_kwargs"]
    gmsh_options = config["gmsh_options"]

    # Select files
    Tk().withdraw()
    filename = askopenfilename(title="Select outer mesh")
    if filename == "":
        return
    Tk().destroy()

    Tk().withdraw()
    filenames = askopenfilenames(title="Select inner meshes")
    if len(filenames) == 0:
        return
    Tk().destroy()

    # Load files
    outer_mesh = pv.PolyData(pv.read(filename))
    inner_meshes = [pv.PolyData(pv.read(filename)) for filename in filenames]

    # Run tetrahedralizer
    combined = preprocess_and_tetrahedralize(outer_mesh, inner_meshes, mesh_repair_kwargs, gmsh_options)

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    filename = f"{output_directory}/{output_filename}{output_extension}"
    combined.save(f"{filename}")

    # Plot result
    p = pv.Plotter()
    cmap = cm.get_cmap("Accent")
    p.add_mesh(combined, opacity=0.15, cmap=cmap, show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = combined.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", cmap=cmap, show_edges=True)

    p.add_plane_widget(plane_func, assign_to_axis="z")
    p.add_title("Combined Tetrahedralized Lung Sections")
    p.show()


if __name__ == "__main__":
    main()
