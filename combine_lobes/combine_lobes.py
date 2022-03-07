import pyvista as pv
import pymeshfix
import numpy as np
import os
import pathlib
import meshio
from numpy.typing import ArrayLike

"""
Testing methods to combine surface meshes of lung lobes. The lobes have coinciding walls, which we want to remove,
then fix the mesh so it is manifold
"""

filenames = ["meshes/LLL.stl", "meshes/LUL.stl"]
output_directory = "output"
output_suffix = "shared_faces_removed"


def main():
    meshes = [pv.read(filename) for filename in filenames]

    p = pv.Plotter()
    for mesh in meshes:
        p.add_mesh(mesh, style="wireframe")
    p.add_title("Input Meshes")
    p.show()

    # #Show in subplots to prove that each lobe has a copy of the wall
    # p = pv.Plotter(shape=(1, len(meshes)))
    # for i, mesh in enumerate(meshes):
    #     p.subplot(0, i)
    #     p.add_mesh(mesh, style="wireframe")
    # p.show()

    # """
    # Attempting to use the intersecting triangles function of pymeshfix. This is not working and seems to return
    # random cells. Pyvista boolean operation are not working either and just error.
    # """
    # blocks = pv.MultiBlock(meshes)
    # combined = blocks.combine()
    # tin = pymeshfix.PyTMesh()
    # tin.load_array(combined.points, combined.cells.reshape(-1, 4)[:, 1:])
    # intersecting = tin.select_intersecting_triangles()
    # cells = combined.cells.reshape(-1, 4)[:, 1:]
    # cells_updated = np.delete(cells, intersecting, 0)
    # mesh_updated = pv.PolyData(combined.points, np.insert(cells_updated, 0, values=3, axis=1).ravel())

    """
    Let's try finding all the faces that use 3 shared nodes.

    This works!
    """
    # TODO make it work with three lobes, put algorithm in a library, make the rest an app

    # Find shared nodes
    tolerance = 1e-05
    # Do this for each pair I guess. Right now this only works if there are only two meshes


    # # Plot shared points. Should appear as overlapping pairs, so with opacity 0.5 the combined color should indicate
    # # the correct presence of 2 overlapping points.
    # p = pv.Plotter()
    # p_a = pv.PolyData(meshes[0].points[np.asarray(shared_points)[:, 0]])
    # p_b = pv.PolyData(meshes[1].points[np.asarray(shared_points)[:, 1]])
    # p.add_points(p_a, color="red", point_size=10, opacity=0.5)
    # p.add_points(p_b, color="blue", point_size=10, opacity=0.5)
    # p.show()

    # Find elements where all nodes are in the shared nodes list


    # # Plot shared faces
    # p = pv.Plotter()
    # for i, mesh in enumerate(meshes):
    #     faces = mesh.faces.reshape(-1, 4)[:, 1:]
    #     faces_trimmed = faces[shared_faces[i]]
    #     shared_face_mesh = pv.PolyData(mesh.points, np.insert(faces_trimmed, 0, values=3, axis=1).ravel())
    #     p.add_mesh(shared_face_mesh, style="wireframe")
    # p.add_title("Shared Faces")
    # p.show()


    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    m = meshio.Mesh(new_mesh.points, {"triangle": new_mesh.faces.reshape(-1, 4)[:, 1:]})
    m.write(f"{output_directory}/{output_filename}.stl")

    p = pv.Plotter()
    p.add_mesh(new_mesh, style="wireframe")
    if holes.number_of_points != 0:
        p.add_mesh(holes, color="r")
    p.add_title("Shared Faces Removed and Mesh Fixed")
    p.show()


def merge_surfaces_removing_shared_faces(meshes: list[pv.PolyData], tolerance: float = None) -> pv.PolyData:
    """
    Merge PyVista Polydata surface meshes and remove faces that any two surfaces share.

    Parameters
    ----------
    meshes
    tolerance

    Returns
    -------

    """
    # For each pair:
    select_shared_faces(mesh_a, meshes, tolerance)
    # Delete shared faces
    shared_faces_removed = []
    for i, mesh in enumerate(meshes):
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        faces_trimmed = np.delete(faces, shared_faces[i], 0)
        shared_face_removed_mesh = pv.PolyData(mesh.points, np.insert(faces_trimmed, 0, values=3, axis=1).ravel())
        shared_faces_removed.append(shared_face_removed_mesh)

    # Combine meshes and use meshfix.repair
    blocks = pv.MultiBlock(shared_faces_removed)
    combined = blocks.combine(merge_points=True, tolerance=1e-05)
    # p = pv.Plotter()
    # p.add_mesh(combined, style="wireframe")
    # p.add_title("Shared Faces Removed")
    # p.show()
    meshfix = pymeshfix.MeshFix(combined.points, combined.cells.reshape(-1, 4)[:, 1:])
    holes = meshfix.extract_holes()
    meshfix.repair()

    # Todo: This loses all face data! how can we do this better
    new_mesh = pv.PolyData(meshfix.v, np.insert(meshfix.f, 0, values=3, axis=1).ravel())

    # Delete faces
    # Check which points are still used, delete the rest, (Could use pv.clean but that could delete other orphaned
    # points which we don't necessarily want to do.
    return new_mesh


def select_shared_faces(mesh_a: pv.PolyData, mesh_b: pv.PolyData, tolerance: float = None) -> tuple:
    shared_points_kwargs = {"mesh_a": mesh_a, "mesh_b": mesh_b, "tolerance": tolerance}
    shared_points = select_shared_points(**{k: v for k, v in shared_points_kwargs.items() if v is not None})

    mesh_a_faces = select_faces_using_points(mesh_a, shared_points[0])
    mesh_b_faces = select_faces_using_points(mesh_b, shared_points[1])

    return mesh_a_faces, mesh_b_faces


def select_shared_points(mesh_a: pv.PolyData, mesh_b: pv.PolyData, tolerance: float = 1e-05) \
        -> tuple[list[ArrayLike],list[ArrayLike]]:
    shared_points = ([],[])
    for i_a, point_a in enumerate(mesh_a.points):
        for i_b, point_b in enumerate(mesh_b.points):
            # linalg.norm calculates euclidean distance
            if np.linalg.norm(point_a - point_b) <= tolerance:
                # Need to remember the index of the shared point in both meshes so we can find faces that use it in both
                shared_points[0].append(i_a)
                shared_points[1].append(i_b)

    return shared_points


def select_faces_using_points(mesh, points: list[tuple[ArrayLike, ArrayLike]]) -> list:
    mesh_faces = []
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    mesh_shared_points_set = set(np.asarray(points)[:, i])
    for j, face in enumerate(faces):
        # Check if all of face are in the current meshes shared points
        if set(face).issubset(mesh_shared_points_set):
            mesh_faces.append(j)

    return mesh_faces


if __name__ == "__main__":
    main()


