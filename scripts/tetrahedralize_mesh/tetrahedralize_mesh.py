import pyvista as pv
import tetgen
import pymeshfix
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import json
import pathlib
import os
import meshio

"""
App to convert a surface mesh to a tetrahedral mesh.
"""

config_filename = "conf.json"


def main():
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to tetrahedralize")
    if filename == "":
        return
    Tk().destroy()

    with open(config_filename, "r") as f:
        config = json.load(f)

    mesh = pv.read(filename)

    tet = tetgen.TetGen(mesh)
    meshfix = pymeshfix.MeshFix(tet.v, tet.f)
    meshfix.repair()
    tet.v, tet.f = meshfix.v, meshfix.f
    tet.tetrahedralize(switches=config["tetrahedralize_switches"])

    path = pathlib.Path(filename)
    output_suffix = config["output_suffix"]
    output_directory = config["output_directory"]

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # TetGen cells should always have four nodes. We reshape and remove the "padding" from PyVista
    m = meshio.Mesh(tet.grid.points, {"quad": tet.grid.cells.reshape(-1, 5)[:, 1:]})
    m.write(f"{output_directory}/{path.stem}_{output_suffix}.ply")

    if config["plot_result"]:
        p = pv.Plotter()

        # p.add_mesh(tet.grid.extract_all_edges())

        # p.add_mesh_clip_plane(tet.grid, invert=True, assign_to_axis=config["clip_axis"], show_edges=True, opacity=0.5)

        def plane_func(normal, origin):
            slc = tet.grid.slice(normal=normal, origin=origin)
            p.add_mesh(slc, name="slice", style="wireframe")

        p.add_mesh(tet.grid, opacity=0.5, show_edges=True, edge_color="gray")
        p.add_plane_widget(plane_func, assign_to_axis=config["clip_axis"])

        p.show()


if __name__ == "__main__":
    main()
