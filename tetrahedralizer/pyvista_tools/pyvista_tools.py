import itertools
from typing import Union, List, Dict, Tuple

import numpy as np
import pyvista
import vtkmodules.util
from numpy.typing import NDArray
import pyvista as pv
from pyvista import UnstructuredGrid


def remove_shared_faces(meshes: List[pv.DataSet], tolerance: float = None,
                        return_removed_faces: bool = False, merge_result=True) -> Union[
        List[pv.PolyData], Tuple[List[pv.PolyData], list]]:
    """
    Remove faces shared by any two of a list of Pyvista Polydata and merge the result. This is similar to the Pyvista
    boolean union, but works with intersections of zero volume. The meshes can optionally be returned unmerged. The
    removed faces can also optionally be returned.

    Parameters
    ----------
    meshes
        List of meshes to merge
    tolerance
        Tolerance for selecting shared points
    merge_result
        If true, returns a list with intersecting meshes merged. Otherwise returns a list of each input individually.
        Default True
    return_removed_faces
        If true, returns a list of faces that were removed.

    Returns
    -------
    List of pv.Polydata with each set of intersecting input meshes merged into one. Input meshes that don't intersect
    any other are returned unchanged. Alternatively, a list of non-merged input meshes with shared faces removed.

    Optionally, list of faces that were removed.

    """
    # For each pair:
    faces_to_remove = [[] for _ in range(len(meshes))]
    intersection_sets = []
    for (index_a, mesh_a), (index_b, mesh_b) in itertools.combinations(enumerate(meshes), 2):
        shared_points_kwargs = {"mesh_a": mesh_a, "mesh_b": mesh_b, "tolerance": tolerance}
        shared_points_a, shared_points_b = select_shared_points(**{k: v for k, v in shared_points_kwargs.items() if v is not None})

        mesh_a_faces = select_faces_using_points(mesh_a, shared_points_a)
        mesh_b_faces = select_faces_using_points(mesh_b, shared_points_b)

        faces_to_remove[index_a].extend(np.array(mesh_a_faces))
        faces_to_remove[index_b].extend(np.array(mesh_b_faces))

        # If mesh_a and mesh_b intersect, we need to record this so we can merge them later
        if (mesh_a_faces, mesh_b_faces) != ([], []):
            # If a is already part of a set, add b to it
            if np.any([index_a in s for s in intersection_sets]):
                intersection_sets[np.argmax([index_a in s for s in intersection_sets])].add(index_b)
            # Else if b is already part of a set, add a to it
            elif np.any([index_b in s for s in intersection_sets]):
                intersection_sets[np.argmax([index_b in s for s in intersection_sets])].add(index_a)
            # Else make a new one with both
            else:
                intersection_sets.append(set([index_a, index_b]))

    trimmed_meshes = []
    for i, mesh in enumerate(meshes):
        if len(faces_to_remove[i]) > 0:
            trimmed = mesh.remove_cells(faces_to_remove[i])
            trimmed_meshes.append(trimmed)
        else:
            trimmed_meshes.append(mesh.copy())

    if merge_result:
        output = []
        # Merge all sets of intersecting meshes and add to output
        for intersection_set in intersection_sets:
            set_list = list(intersection_set)
            merged = pv.PolyData()
            for index in set_list:
                merged = merged.merge(trimmed_meshes[index], merge_points=True)
            output.append(merged)

        # Add any that were not part of an intersection set
        intersecting_indices = list(itertools.chain(*intersection_sets))
        for index, mesh in enumerate(trimmed_meshes):
            if index not in intersecting_indices:
                output.append(mesh)

    else:
        output = trimmed_meshes

    if not return_removed_faces:
        return output
    else:
        return output, faces_to_remove


def select_shared_faces(mesh_a: pv.PolyData, mesh_b: pv.PolyData, tolerance: float = None) -> Tuple[list, list]:
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


def select_points_in_faces(mesh: pv.PolyData, points: List[int] = None, faces: List[int] = None,
                           exclusive: bool = False) -> List[int]:
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


def pyvista_faces_by_dimension(faces: NDArray) -> Dict[int, NDArray]:
    output = {}
    i = 0
    while i < len(faces):
        # Preceding each face is the "padding" indicating the number of elements in the face
        num_elems = faces[i]
        # Append padding plus each element to the output dict
        if num_elems in output:
            output[num_elems] = np.append(output[num_elems], np.array([faces[i+j] for j in range(num_elems+1)]))
        else:
            output[num_elems] = np.array([faces[i+j] for j in range(num_elems+1)])
        # Increment index to the next padding number
        i += num_elems + 1
    return output


def pyvista_faces_to_2d(faces: NDArray) -> NDArray:
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


def pyvista_faces_to_1d(faces: NDArray) -> NDArray:
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
        -> Tuple[List[int], List[int]]:
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


def select_faces_using_points(mesh: pv.PolyData, points: list) -> List[int]:
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


def pyvista_tetrahedral_mesh_from_arrays(nodes, tets) -> pyvista.UnstructuredGrid:
    cell_type = np.hstack([
        np.ones(len(tets)) * vtkmodules.util.vtkConstants.VTK_TETRA
    ])
    mesh = pv.UnstructuredGrid(pyvista_faces_to_1d(tets), cell_type, nodes)
    return mesh