import itertools
import numpy as np
import os
import pathlib
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import gmsh
import json
import pyvista as pv

config_filename = "conf.json"

def main():
    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to gmsh")
    if len(filenames) == 0:
        return
    Tk().destroy()

    with open(config_filename, "r") as f:
        config = json.load(f)

    output_directory = config["output_directory"]
    output_suffix = config["output_suffix"]

    gmsh.initialize()

    for filename in filenames:
        gmsh.merge(filename)

    # Create a volume from all the surfaces
    s = gmsh.model.get_entities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])

    gmsh.model.geo.synchronize()

    for name, value in config["gmsh_options"].items():
        gmsh.option.set_number(name, value)

    gmsh.model.mesh.generate(3)

    # Save result
    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    gmsh.write(f"{output_directory}/{output_filename}.msh")



    # # Plot result
    # p = pv.Plotter()
    #
    # p.add_mesh(combined, opacity=0.15, show_edges=True, edge_color="gray")
    #
    # def plane_func(normal, origin):
    #     slc = combined.slice(normal=normal, origin=origin)
    #     p.add_mesh(slc, name="slice", show_edges=True)
    #
    # p.add_plane_widget(plane_func, assign_to_axis="z")
    #
    # p.add_title("Combined Tetrahedralized Lung Sections")
    # p.show()

    gmsh.finalize()



if __name__ == "__main__":
    main()
