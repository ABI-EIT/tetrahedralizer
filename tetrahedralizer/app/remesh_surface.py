import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pyacvd
import pathlib
import os

"""
App to remesh a surface using pyacvd
"""


def main():
    remesh_divs = 5
    nclus = 3000
    output_directory = "output"
    output_suffix = "remeshed"
    output_extension = ".stl"

    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to remesh")
    if filename == "":
        return
    Tk().destroy()

    mesh = pv.read(filename)

    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True)
    p.add_title("Input Mesh")
    p.add_axes_at_origin()
    p.show()

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(remesh_divs)
    clus.cluster(nclus)
    clus.plot()

    remesh = clus.create_mesh()
    p = pv.Plotter()
    p.add_mesh(remesh, show_edges=True)
    p.add_title("Remeshed Mesh")
    p.add_axes_at_origin()
    p.show()

    # Save
    path = pathlib.Path(filename)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    remesh.save(f"{output_directory}/{path.stem}_{output_suffix}{output_extension}")


if __name__ == "__main__":
    main()
