import pyvista as pv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import cm
import numpy as np
from tetrahedralizer.pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d

"""

"""

def main():
    Tk().withdraw()
    filename = askopenfilename(title="Select mesh to view")
    if filename == "":
        return
    Tk().destroy()

    mesh = pv.read(filename)

    lobe_names = ["lower_lobe_of_left_lung", "upper_lobe_of_left_lung", "lower_lobe_of_right_lung",
                  "middle_lobe_of_right_lung", "upper_lobe_of_right_lung"]

    all_lobe_data = np.zeros(mesh.n_cells)
    for i, name in enumerate(lobe_names):
        all_lobe_data += (i+1) * mesh.cell_data[name]

    mesh.cell_data["all_lobes"] = all_lobe_data

    lll_cells = pyvista_faces_to_2d(mesh.cells)[mesh.cell_data["all_lobes"] == 1]
    lll_mesh = pv.UnstructuredGrid({12: lll_cells}, mesh.points)
    surface = lll_mesh.extract_surface()
    p = pv.Plotter()
    cmap = cm.get_cmap("Accent")
    p.add_mesh(mesh, opacity=0.3, cmap=cmap, scalars=mesh.cell_data["all_lobes"], show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = mesh.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", scalars=slc.cell_data["all_lobes"], cmap=cmap, show_edges=True)

    p.add_title("Lung Scaffold with Colored Lobes")
    p.add_plane_widget(plane_func, assign_to_axis="z")
    p.show()


if __name__ == "__main__":
    main()
