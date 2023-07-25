import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np

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
    cmap = colormaps["Accent"]
    p.add_mesh(mesh, opacity=0.15, cmap=cmap, show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = mesh.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", cmap=cmap, show_edges=True)

    p.add_plane_widget(plane_func, assign_to_axis="y")
    p.show()

    aspect_ratio = mesh.compute_cell_quality(quality_measure="aspect_ratio")["CellQuality"]
    volume = mesh.compute_cell_quality(quality_measure="volume")["CellQuality"]

    fig, axs = plt.subplots(1, 2)
    axs[0].hist(volume, bins=np.linspace(np.min(volume), np.max(volume) + 1, 100), density=True)
    axs[0].title.set_text("Cell Volume Frequency")
    axs[0].set_xlabel("Cell Volume")
    axs[0].set_ylabel("Probability Density")
    axs[1].hist(aspect_ratio, bins=np.linspace(np.min(aspect_ratio), np.max(aspect_ratio) + 1, 100), density=True)
    axs[1].title.set_text("Cell Aspect Ratio Frequency")
    axs[1].set_xlabel("Cell Aspect Ratio")
    axs[1].set_ylabel("Probability Density")
    fig.set_size_inches((10, 4))
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
