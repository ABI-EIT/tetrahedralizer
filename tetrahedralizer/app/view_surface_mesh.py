import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import cm

"""
App to view a mesh using pyvista
"""

def main():
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to view")
    if filename == "":
        return
    Tk().destroy()

    mesh = pv.read(filename)

    plot_with_non_manifold(mesh)

def plot_with_non_manifold(mesh: pv.PolyData, title=None):
    p = pv.Plotter()
    nm = mesh.extract_feature_edges(feature_edges=False, boundary_edges=False,
                                    manifold_edges=False, non_manifold_edges=True)

    p.add_mesh(mesh, color="white", style="wireframe", label="Surface Mesh")

    if nm.n_points != 0:
        p.add_mesh(nm, color="red", label="Non-Manifold Edges")
    else:
        print("No non-manifold edges")

    p.add_legend()
    if title is not None:
        p.add_title(title)
    p.show()


if __name__ == "__main__":
    main()
