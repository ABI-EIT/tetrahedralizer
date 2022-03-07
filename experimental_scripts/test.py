import pyvista as pv
import tetgen
import pymeshfix
import numpy as np

filenames = ["meshes/torso.stl", "meshes/LLL.stl", "meshes/LUL.stl",
             "meshes/RLL.stl", "meshes/RML.stl", "meshes/RUL.stl"]
file_elem_values = [1, 2, 2, 2, 2, 2]


def main():
    meshes = [pv.read(filename) for filename in filenames]

    # p = pv.Plotter()
    # for mesh in meshes:
    #     p.add_mesh(mesh, style="wireframe")
    # p.show()

    # Repair torso because it is formed open at the top and bottom
    meshfix = pymeshfix.MeshFix(meshes[0].points, meshes[0].faces.reshape(-1, 4)[:, 1:])
    meshfix.repair()
    meshes[0].points = meshfix.v
    meshes[0].faces = np.insert(meshfix.f, 0, values=3, axis=1).ravel()

    blocks = pv.MultiBlock(meshes)
    combined = blocks.combine()
    combined.plot(style="wireframe")

    tet = tetgen.TetGen(combined.points, combined.cells.reshape(-1, 4)[:, 1:])
    tet.tetrahedralize()
    tet.grid.plot(style="wireframe")


    # tet = tetgen.TetGen(mesh)
    # meshfix = pymeshfix.MeshFix(tet.v, tet.f)
    # holes = meshfix.extract_holes()
    # meshfix.repair()
    # tet.v, tet.f = meshfix.v, meshfix.f
    # tet.tetrahedralize(switches="a100")
    #
    # # p = pv.Plotter()
    # # p.add_mesh(mesh, style="wireframe", color="k", label="Surface")
    # # p.add_mesh(holes, color='r', label="holes")
    # # p.add_mesh(meshfix.mesh, label="Repaired surface", opacity=0.75)
    # # p.add_legend()
    # # p.add_axes()
    # # p.show()
    #
    # p = pv.Plotter()
    # p.add_mesh_clip_plane(tet.grid, invert=True, assign_to_axis="z", show_edges=True)
    # p.show()


if __name__ == "__main__":
    main()
