import pyvista as pv
import numpy as np
import os
import pathlib
import meshio
import itertools
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
"""
Testing methods to combine surface meshes of lung lobes. The lobes have coinciding walls, which we want to remove,
then fix the mesh so it is manifold
"""

output_directory = "output"
output_suffix = "shared_faces_removed"


def main():

    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to merge")
    if len(filenames) == 0:
        return
    Tk().destroy()
    meshes = [pv.PolyData(pv.read(filename)) for filename in filenames]

    p = pv.Plotter()
    for mesh in meshes:
        p.add_mesh(mesh, style="wireframe")
    p.add_title("Input Meshes")
    p.show()

    combined = merge_surfaces_removing_shared_faces(meshes)

    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    m = meshio.Mesh(combined.points, {"triangle": pyvista_faces_to_2d(combined.cells)})
    m.write(f"{output_directory}/{output_filename}.stl")

    p = pv.Plotter()
    p.add_mesh(combined, style="wireframe")
    p.add_title("Shared Faces Removed")
    p.show()


def merge_surfaces_removing_shared_faces(meshes: list[pv.PolyData], tolerance: float = None) -> pv.UnstructuredGrid:
    """
    Merge PyVista Polydata surface meshes and remove faces that any two surfaces share.

    Parameters
    ----------
    meshes
        List of meshes to merge
    tolerance
        Tolerance for selecting shared points

    Returns
    -------
    Unstructured grid representing the combination of the input meshes, with shared walls removed.

    """
    # For each pair:
    for mesh_a, mesh_b in itertools.combinations(meshes, 2):
        shared_points_kwargs = {"mesh_a": mesh_a, "mesh_b": mesh_b, "tolerance": tolerance}
        shared_points_a, shared_points_b = select_shared_points(**{k: v for k, v in shared_points_kwargs.items() if v is not None})

        mesh_a_faces = select_faces_using_points(mesh_a, shared_points_a)
        mesh_b_faces = select_faces_using_points(mesh_b, shared_points_b)

        to_delete_mesh_a = select_points_in_faces(mesh_a, shared_points_a, mesh_a_faces, exclusive=True)
        to_delete_mesh_b = select_points_in_faces(mesh_b, shared_points_b, mesh_b_faces, exclusive=True)

        if len(to_delete_mesh_a) > 0:
            mesh_a.remove_points(to_delete_mesh_a, inplace=True)
        if len(to_delete_mesh_b) > 0:
            mesh_b.remove_points(to_delete_mesh_b, inplace=True)

    blocks = pv.MultiBlock(meshes)
    combined = blocks.combine(merge_points=True, tolerance=1e-05)

    return combined


# Not used any more because we need to use the shared points too.
def select_shared_faces(mesh_a: pv.PolyData, mesh_b: pv.PolyData, tolerance: float = None) -> tuple[list, list]:
    """
    Select that faces that two meshes share. Shared faces are determined by selecting the faces that use shared points.

    Parameters
    ----------
    mesh_a
        First mesh
    mesh_b
        Second mesh
    tolerance
        Tolerance for selecting shared points

    Returns
    -------
    Tuple of lists containing the indices of the shared faces in mesh_a and mesh_b

    """
    shared_points_kwargs = {"mesh_a": mesh_a, "mesh_b": mesh_b, "tolerance": tolerance}
    shared_points = select_shared_points(**{k: v for k, v in shared_points_kwargs.items() if v is not None})

    mesh_a_faces = select_faces_using_points(mesh_a, shared_points[0])
    mesh_b_faces = select_faces_using_points(mesh_b, shared_points[1])

    return mesh_a_faces, mesh_b_faces


def select_points_in_faces(mesh: pv.PolyData, points: list[int] = None, faces: list[int] = None,
                           exclusive: bool = False) -> list[int]:
    """
    Select points used in the faces of a mesh. Optionally specify a subset of points and/or faces to check. When
    exclusive is set to True, selects only points that are not used in the remaining faces.

    Only works on meshes with all the same number of points per face.

    Parameters
    ----------
    mesh
        Mesh to select from
    points
        Optional subset of points in mesh to select from
    faces
        Optional subset of faces in mesh to test
    exclusive
        If true, selects only points exclusively used in the specified faces

    Returns
    -------
    List of points used in the specified faces

    """
    mesh_faces_2d = pyvista_faces_to_2d(mesh.faces)
    faces_2d = mesh_faces_2d[faces]
    remaining_faces = np.delete(mesh_faces_2d, faces, 0)

    # Check if any given point exists in the faces to search
    used_in_faces = []
    for point in points:
        if any(point in face for face in faces_2d):
            used_in_faces.append(point)

    # If exclusive is set, remove points that also exist in the remaining faces
    excluded_points = []
    if exclusive:
        for point in used_in_faces:
            if any(point in face for face in remaining_faces):
                excluded_points.append(point)

    used_in_faces = list(set(used_in_faces) - set(excluded_points))

    return used_in_faces


def pyvista_faces_to_2d(faces: np.ndarray) -> np.ndarray:
    """
    Convert pyvista faces from the native 1d array to a 2d array with one face per row. Padding is trimmed.

    Only works on a list of faces with all the same number of points per face.

    Parameters
    ----------
    faces
        Faces to be reshaped

    Returns
    -------
    2d array of faces
    """
    points_per_face = faces[0]
    return faces.reshape(-1, points_per_face+1)[:, 1:]


def pyvista_faces_to_1d(faces: np.ndarray) -> np.ndarray:
    """
    Convert 2d array of faces to the pyvista native 1d format, inserting the padding.

    Only works on a list of faces with all the same number of points per face.

    Parameters
    ----------
    faces
        Faces to be reshaped

    Returns
    -------
    1d array of faces

    """
    padding = len(faces[0])
    return np.insert(faces, 0, values=padding, axis=1).ravel()


def select_shared_points(mesh_a: pv.PolyData, mesh_b: pv.PolyData, tolerance: float = 1e-05) \
        -> tuple[list[int], list[int]]:
    """
    Select the points that two meshes share. Points are considered shared if they are within a specified euclidian
    distance from one another.

    Parameters
    ----------
    mesh_a
        First mesh
    mesh_b
        Second mesh
    tolerance
        Maximum euclidian distance between points to consider them shared

    Returns
    -------
    shared_points
        Tuple containing indices of shared points in mesh_a, and shared points in mesh_b

    """
    shared_points_a = []
    shared_points_b = []
    for i_a, point_a in enumerate(mesh_a.points):
        for i_b, point_b in enumerate(mesh_b.points):
            # linalg.norm calculates euclidean distance
            if np.linalg.norm(point_a - point_b) <= tolerance:
                # Need to remember the index of the shared point in both meshes so we can find faces that use it in both
                shared_points_a.append(i_a)
                shared_points_b.append(i_b)

    return shared_points_a, shared_points_b


def select_faces_using_points(mesh: pv.PolyData, points: list) -> list[int]:
    """
    Select all faces in a mesh that contain only the specified points.

    Parameters
    ----------
    mesh:
        Mesh to select from
    points
        The only points in the given mesh that the selected faces may contain

    Returns
    -------
    mesh_faces
        List of indices of the selected faces in the mesh

    """
    mesh_faces = []
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    points_set = set(points)
    for j, face in enumerate(faces):
        # Check if all of the points in each face are in the target points set
        if set(face).issubset(points_set):
            mesh_faces.append(j)

    return mesh_faces


if __name__ == "__main__":
    main()


