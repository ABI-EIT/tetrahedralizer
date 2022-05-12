import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pyvista_tools import extract_enclosed_regions
import numpy as np
from matplotlib import cm


def main():
    # Select File
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to extract regions")
    if filename == "":
        return
    Tk().destroy()

    # Load File
    mesh = pv.read(filename)

    regions = extract_enclosed_regions(mesh)
    cmap = cm.get_cmap("Set1")
    p = pv.Plotter()
    for i, region in enumerate(regions):
        p.add_mesh(region, style="wireframe", color=cmap(i))

    p.show()

    main_region = regions[np.argmax([region.volume for region in regions])]
    plot_with_non_manifold(main_region)


def plot_with_non_manifold(mesh: pv.PolyData, title=None):
    p = pv.Plotter()
    nm = mesh.extract_feature_edges(feature_edges=False, boundary_edges=False,
                                    manifold_edges=False, non_manifold_edges=True)

    p.add_mesh(mesh, color="white", style="wireframe", label="Surface Mesh")
    p.add_mesh(nm, color="red", label="Non-Manifold Edges")
    p.add_legend()
    if title is not None:
        p.add_title(title)
    p.show()


if __name__ == "__main__":
    main()
