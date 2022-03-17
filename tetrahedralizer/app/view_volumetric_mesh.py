import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import cm

"""
App to view a mesh using the pyvista plane widget
"""

def main():
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to view")
    if filename == "":
        return
    Tk().destroy()

    mesh = pv.read(filename)

    p = pv.Plotter()
    cmap = cm.get_cmap("Accent")
    p.add_mesh(mesh, opacity=0.15, cmap=cmap, show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = mesh.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", cmap=cmap, show_edges=True)

    p.add_plane_widget(plane_func, assign_to_axis="z")
    p.show()


if __name__ == "__main__":
    main()
