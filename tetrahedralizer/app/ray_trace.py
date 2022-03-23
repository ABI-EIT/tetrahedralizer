import pyvista as pv
import itertools
import numpy as np

from tkinter import Tk
from tkinter.filedialog import askopenfilenames
filenames = [r"..\..\examples\meshes\mock_lung\mock_LUL.STL", r"..\..\examples\meshes\mock_lung\mock_LLL.STL"]
tolerance = 0.1
merge_result = True


def main():

    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to merge")
    if len(filenames) == 0:
        return
    Tk().destroy()

    meshes = [pv.PolyData(filename) for filename in filenames]

    # Construct rays normal to each mesh face of length 1*tolerance
    mesh_rays = []
    for mesh in meshes:
        mesh.compute_normals()
        ray_starts = mesh.cell_centers().points
        ray_ends = + ray_starts + mesh.cell_normals*tolerance
        mesh_rays.append([(ray_start, ray_end) for ray_start, ray_end in zip(ray_starts, ray_ends)])

    cells_to_remove = [[] for _ in range(len(meshes))]
    for (i_a, (mesh_a, mesh_rays_a)), (i_b, (mesh_b, mesh_rays_b)) in itertools.combinations(enumerate(zip(meshes, mesh_rays)), 2):
        # Check which rays from mesh b hit mesh a
        _, inds = zip(*[mesh_a.ray_trace(*ray) for ray in mesh_rays_b])
        b_hit_something = []
        for i, ind in enumerate(inds):
            if len(ind) > 0:
                b_hit_something.append(i)

        # Any rays that hit something correspond to cells that should be removed
        if b_hit_something:
            cells_to_remove[i_b].extend(np.array(b_hit_something))

        # Check which rays from mesh a hit mesh b
        _, inds = zip(*[mesh_b.ray_trace(*ray) for ray in mesh_rays_a])
        a_hit_something = []
        for i, ind in enumerate(inds):
            if len(ind) > 0:
                a_hit_something.append(i)

        # Any rays that hit something correspond to cells that should be removed
        if a_hit_something:
            cells_to_remove[i_a].extend(np.array(a_hit_something))

    for i, mesh in enumerate(meshes):
        mesh.remove_cells(cells_to_remove[i], inplace=True)

    trimmed_meshes = []
    for i, mesh in enumerate(meshes):
        if len(cells_to_remove[i]) > 0:
            trimmed = mesh.remove_cells(cells_to_remove[i])
            trimmed_meshes.append(trimmed)
        else:
            trimmed_meshes.append(mesh.copy())

    if merge_result:
        output = pv.PolyData()
        for mesh in trimmed_meshes:
            output = output.merge(mesh, merge_points=True)

        p = pv.Plotter()
        for mesh in meshes:
            p.add_mesh(mesh, style="wireframe")
        p.show()


    else:
        output = trimmed_meshes






if __name__ == "__main__":
    main()
