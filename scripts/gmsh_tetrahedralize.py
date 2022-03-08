import itertools

import numpy as np
import os
import pathlib
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import gmsh

output_directory = "output"
output_suffix = "gmshed"


def main():
    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to gmsh")
    if len(filenames) == 0:
        return
    Tk().destroy()

    gmsh.initialize()

    for filename in filenames:
        gmsh.merge(filename)

    # Create a volume from all the surfaces
    s = gmsh.model.get_entities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])
    #
    # for a, b in itertools.combinations(gmsh.model.get_entities(2), 2):
    #     fragment = gmsh.model.occ.fragment(a, b)

    # elements = gmsh.model.mesh.get_elements(2)

    gmsh.model.geo.synchronize()

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

    gmsh.finalize()


if __name__ == "__main__":
    main()
