import itertools
from typing import Union, List, Dict, Tuple

import numpy as np
import pyvista
import vtkmodules.util
from numpy.typing import NDArray
import pyvista as pv
from pyvista import UnstructuredGrid
from tqdm import tqdm


def remove_shared_faces_with_ray_trace(meshes: List[pv.DataSet], ray_length: float = 0.01,
                                       incidence_angle_tolerance: float = 0.01,
                                       return_removed_faces: bool = False, merge_result=True)\
                                                            -> Union[List[pv.PolyData], Tuple[List[pv.PolyData], list]]:
    """

    Parameters
    ----------
    meshes
    ray_length
    incidence_angle_tolerance
    return_removed_faces
    merge_result

    Returns
    -------

    """
    # Construct rays normal to each mesh face of length 1*ray_length
    mesh_rays = []
    for mesh in meshes:
        mesh = mesh.compute_normals(auto_orient_normals=True)
        ray_starts = mesh.cell_centers().points - (mesh.cell_normals * ray_length)
        ray_ends = mesh.cell_centers().points + (mesh.cell_normals * ray_length)
        # Create list of tuples, each corresponding to a single ray
        mesh_rays.append([(ray_start, ray_end) for ray_start, ray_end in zip(ray_starts, ray_ends)])

    cells_to_remove = [[] for _ in range(len(meshes))]
    intersection_sets = []
    # Iterate through all permutations with mesh_b shooting rays and mesh_a checking them
    for (i_a, (mesh_a, _)), (i_b, (mesh_b, mesh_rays_b)) in itertools.permutations(enumerate(zip(meshes, mesh_rays)), 2):
        # Check which rays from mesh b hit mesh a
        _, intersection_cells = zip(*[mesh_a.ray_trace(*ray) for ray in mesh_rays_b])

        ray_hits = []
        for i, intersection_cell in enumerate(intersection_cells):
            # If a ray hit a cell, check the angle of incidence
            if len(intersection_cell) > 0:
                # Index of intersection_cells refers to cells in mesh_b. The cell itself refers to cells in mesh_a
                angle_of_indicence = (angle_between(mesh_a.cell_normals[intersection_cell], mesh_b.cell_normals[i]) % (np.pi/2))[0]
                if 0.5 * incidence_angle_tolerance > angle_of_indicence > -0.5 * incidence_angle_tolerance:
                    ray_hits.append(i)

        # Remove cells whose ray hit a parallel cell
        # If mesh_a and mesh_b intersect, we need to record this so we can merge them later
        if ray_hits:
            cells_to_remove[i_b].extend(np.array(ray_hits))

            # If a is already part of a set, add b to it
            if np.any([i_a in s for s in intersection_sets]):
                intersection_sets[np.argmax([i_a in s for s in intersection_sets])].add(i_b)
            # Else if b is already part of a set, add a to it
            elif np.any([i_b in s for s in intersection_sets]):
                intersection_sets[np.argmax([i_b in s for s in intersection_sets])].add(i_a)
            # Else make a new one with both
            else:
                intersection_sets.append(set([i_a, i_b]))

    trimmed_meshes = []
    for i, mesh in enumerate(meshes):
        if len(cells_to_remove[i]) > 0:
            trimmed = mesh.remove_cells(cells_to_remove[i])
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
        return output, cells_to_remove


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def remove_shared_faces(meshes: List[pv.DataSet], tolerance: float = None,
                        return_removed_faces: bool = False, merge_result=True, progress_bar: bool = False) -> Union[
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
    progress_bar
        Include a progress bar

    Returns
    -------
    List of pv.Polydata with each set of intersecting input meshes merged into one. Input meshes that don't intersect
    any other are returned unchanged. Alternatively, a list of non-merged input meshes with shared faces removed.

    Optionally, list of faces that were removed.

    """
    # For each pair:
    faces_to_remove = [[] for _ in range(len(meshes))]
    intersection_sets = []
    for (index_a, mesh_a), (index_b, mesh_b) in tqdm(list(itertools.combinations(enumerate(meshes), 2)),
                                                     disable=not progress_bar, desc="Removing Shared Faces"):
        shared_points_kwargs = {"mesh_a": mesh_a, "mesh_b": mesh_b, "tolerance": tolerance}
        shared_points_a, shared_points_b = select_shared_points(**{k: v for k, v in shared_points_kwargs.items() if v is not None}, progress_bar=progress_bar)

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

    Todo: not crash if you don't specify points (till then you can use list(range(len(mesh.points)))

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


def select_shared_points(mesh_a: pv.PolyData, mesh_b: pv.PolyData, tolerance: float = 1e-05, progress_bar: bool = False) \
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
    progress_bar

    Returns
    -------
    shared_points
        Tuple containing indices of shared points in mesh_a, and shared points in mesh_b

    """
    shared_points_a = []
    shared_points_b = []
    for i_a, point_a in tqdm(list(enumerate(mesh_a.points)), disable= not progress_bar, desc="Selecting Shared Points"):
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
    faces = pyvista_faces_to_2d(mesh.faces)
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


def extract_faces_with_edges(dataset: pv.PolyData, edges: pv.PolyData):

    dataset = dataset.merge(edges)

    faces_using_edges = []
    for i, face in enumerate(pyvista_faces_to_2d(dataset.faces)):
        for line in pyvista_faces_to_2d(dataset.lines):
            if find_sequence(face, line) >= 0:
                faces_using_edges.append(i)

    return faces_using_edges


def find_sequence(array, sequence):
    location = -1
    # hstack array so we can find sequences that wrap around
    array = np.hstack((array, array))
    for i in range(len(array)-len(sequence)+1):
        if np.all(array[i:i+len(sequence)] == sequence):
            location = i
            break
    return location


def compute_face_winding_orders(mesh: pv.PolyData):
    if not mesh.is_all_triangles:
        raise ValueError("Mesh must be all triangles")

    faces = pyvista_faces_to_2d(mesh.faces)
    face_coords = mesh.points[faces]

    winding_orders = []
    for coords, normal in zip(face_coords, mesh.face_normals):
        winding_order = compute_triangle_winding_order(coords[0], coords[1], coords[2], normal)
        winding_orders.append(winding_order)

    return winding_orders


def compute_triangle_winding_order(a, b, c, normal):
    expected_normal = np.cross(b - a, c - b)
    agreement = np.dot(expected_normal, normal)

    return agreement


def rewind_face(mesh, face_num, inplace=False):
    faces = pyvista_faces_to_2d(mesh.faces)
    face = faces[face_num]
    face = [face[0], *face[-1:0:-1]]  # Start at same point, then reverse the rest of the face nodes
    faces[face_num] = face

    if inplace:
        mesh.faces = pyvista_faces_to_1d(faces)
    else:
        mesh_out = mesh.copy()
        mesh_out.faces = pyvista_faces_to_1d(faces)
        return mesh_out


def rewind_faces_to_normals(mesh, inplace=False):
    mesh_c = mesh.copy()

    mesh_face_order = compute_face_winding_orders(mesh_c)
    for i in np.where(np.array(mesh_face_order) < 0)[0]:
        mesh_c = rewind_face(mesh_c, i)

    if inplace:
        mesh.faces = mesh_c.faces
    else:
        return mesh_c
