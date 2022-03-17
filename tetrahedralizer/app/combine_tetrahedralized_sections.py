import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import pathlib
import os
import numpy as np
from matplotlib import cm

"""
App combine tetrahedral meshes
"""


def main():
    output_directory = "output"
    output_suffix = "combined"
    output_extension = ".vtu"
    filename_max_length = 200  # Tested on windows. Should be 255 though.
    backup_filename = "tetrahedralized_mesh"

    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to combine")
    if len(filenames) == 0:
        return
    Tk().destroy()

    meshes = [pv.read(filename) for filename in filenames]
    for i, mesh in enumerate(meshes):
        mesh.cell_data["Scalar"] = np.asarray([i % len(meshes)] * mesh.n_cells)

    blocks = pv.MultiBlock(meshes)
    combined = blocks.combine()

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    filename = f"{output_filename}{output_extension}"
    if len(filename) > filename_max_length:
        filename = create_unique_file_name(base=backup_filename, extension=output_extension)
    combined.save(f"{output_directory}/{filename}")

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


def create_unique_file_name(directory=".", base="", extension=""):
    addition = ""
    i = 0
    while True:
        try_name = directory + "/" + base + addition + extension
        if not (os.path.exists(try_name)):
            return try_name
        i += 1
        addition = "_" + str(i)


if __name__ == "__main__":
    main()
