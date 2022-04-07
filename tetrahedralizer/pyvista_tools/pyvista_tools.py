from __future__ import annotations
import itertools
from typing import List, Dict, Tuple

import numpy as np
import pyvista
import vtkmodules.util
from numpy.typing import NDArray, ArrayLike
import pyvista as pv
from pyvista import UnstructuredGrid
from tqdm import tqdm
import pymeshfix

def remove_shared_faces_with_ray_trace(meshes: List[pv.DataSet], ray_length: float = 0.01,
                                       incidence_angle_tolerance: float = 0.01,
                                       return_removed_faces: bool = False, merge_result=True) \
        -> List[pv.PolyData] | Tuple[List[pv.PolyData], list]:
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
    for (i_a, (mesh_a, _)), (i_b, (mesh_b, mesh_rays_b)) in itertools.permutations(enumerate(zip(meshes, mesh_rays)),
                                                                                   2):
        # Check which rays from mesh b hit mesh a
        _, intersection_cells = zip(*[mesh_a.ray_trace(*ray) for ray in mesh_rays_b])

        ray_hits = []
        for i, intersection_cell in enumerate(intersection_cells):
            # If a ray hit a cell, check the angle of incidence
            if len(intersection_cell) > 0:
                # Index of intersection_cells refers to cells in mesh_b. The cell itself refers to cells in mesh_a
                angle_of_indicence = \
                (angle_between(mesh_a.cell_normals[intersection_cell], mesh_b.cell_normals[i]) % (np.pi / 2))[0]
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
                        return_removed_faces: bool = False, merge_result=True, progress_bar: bool = False) -> \
        List[pv.PolyData] | Tuple[List[pv.PolyData], list]:
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
        shared_points_a, shared_points_b = select_shared_points(
            **{k: v for k, v in shared_points_kwargs.items() if v is not None}, progress_bar=progress_bar)

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
    """
    You can also do this by casting to UnstructuredGrid, where the face types are available in a dict

    Parameters
    ----------
    faces

    Returns
    -------

    """
    output = {}
    i = 0
    while i < len(faces):
        # Preceding each face is the "padding" indicating the number of elements in the face
        num_elems = faces[i]
        # Append padding plus each element to the output dict
        if num_elems in output:
            output[num_elems] = np.append(output[num_elems], np.array([faces[i + j] for j in range(num_elems + 1)]))
        else:
            output[num_elems] = np.array([faces[i + j] for j in range(num_elems + 1)])
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
    return faces.reshape(-1, points_per_face + 1)[:, 1:]


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
    for i_a, point_a in tqdm(list(enumerate(mesh_a.points)), disable=not progress_bar, desc="Selecting Shared Points"):
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
    """
    Create a Pyvista Unstructured Grid with tetrahedral cells from an array representation of 3xN nodes and 4xM tets

    Parameters
    ----------
    nodes
    tets

    Returns
    -------

    """
    cell_type = np.hstack([
        np.ones(len(tets)) * vtkmodules.util.vtkConstants.VTK_TETRA
    ])
    mesh = pv.UnstructuredGrid(pyvista_faces_to_1d(tets), cell_type, nodes)
    return mesh


def extract_faces_with_edges(dataset: pv.PolyData, edges: pv.PolyData):
    """
    Extract all the faces of a Pyvista Polydata object that use a given set of edges

    Parameters
    ----------
    dataset
    edges

    Returns
    -------

    """
    dataset = dataset.merge(edges)

    faces_using_edges = []
    for i, face in enumerate(pyvista_faces_to_2d(dataset.faces)):
        for line in pyvista_faces_to_2d(dataset.lines):
            if find_sequence(face, line) >= 0:
                faces_using_edges.append(i)

    return faces_using_edges


def find_sequence(array, sequence, check_reverse=False):
    """
    Find the start index of a subsequence in an array.

    Parameters
    ----------
    array
    sequence

    Returns
    -------
    Location
        -1 represents not found

    """
    location = -1
    # hstack array so we can find sequences that wrap around
    search_array = np.hstack((array, array))
    for i in range(len(search_array) - len(sequence) + 1):
        if np.all(search_array[i:i + len(sequence)] == sequence):
            location = i
            break

    if location == -1 and check_reverse:
        location = find_sequence(array, sequence[::-1], check_reverse=False)

    return location


def compute_face_winding_orders(mesh: pv.PolyData) -> List[float]:
    """
    Compute the face winding orders for an all triangular Pyvista Polydata object with respect to the face normals.

    Parameters
    ----------
    mesh

    Returns
    -------
    winding_orders:
        List of float representing winding order. positive numbers represent positive winding direction with respect to
        the face normal.


    """
    if not mesh.is_all_triangles:
        raise ValueError("Mesh must be all triangles")

    faces = pyvista_faces_to_2d(mesh.faces)
    face_coords = mesh.points[faces]

    winding_orders = []
    for coords, normal in zip(face_coords, mesh.face_normals):
        winding_order = compute_triangle_winding_order(coords[0], coords[1], coords[2], normal)
        winding_orders.append(winding_order)

    return winding_orders


def compute_triangle_winding_order(a, b, c, normal) -> float:
    """
    Compute winding order of a single triangle with respect to the normal

    Parameters
    ----------
    a
    b
    c
    normal

    Returns
    -------

    """
    expected_normal = np.cross(b - a, c - b)
    agreement = np.dot(expected_normal, normal)

    return agreement


def rewind_face(mesh, face_num, inplace=False):
    """
    Reverse the winding direction of a single face of a pyvista polydata

    Parameters
    ----------
    mesh
    face_num
    inplace

    Returns
    -------

    """
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
    """
    Re-order the faces of a Pyvista Polydata to match the face normals

    Parameters
    ----------
    mesh
    inplace

    Returns
    -------

    """
    mesh_c = mesh.copy()

    mesh_face_order = compute_face_winding_orders(mesh_c)
    for i in np.where(np.array(mesh_face_order) < 0)[0]:
        mesh_c = rewind_face(mesh_c, i)

    if inplace:
        mesh.faces = mesh_c.faces
    else:
        return mesh_c


def find_loops_and_chains(lines: ArrayLike):
    """
    Find loops and chains in a list of lines

    Parameters
    ----------
    lines: Nx2 ArrayLike
    """
    edges = []
    for line in lines:
        line_in_loops = [line[0] in itertools.chain(*edge) or line[1] in itertools.chain(*edge) for edge in edges]
        # If either end of the line is already in a loop, add the line to that loop
        if np.any(line_in_loops):
            edges[np.argmax(line_in_loops)].add(tuple(line))
        # Otherwise, start a new loop
        else:
            s = set()
            s.add(tuple(line))
            edges.append(s)

    # Before sorting, classify into loops and chains
    # Loops have all nodes exactly twice. Chains have one line with a unique node 0, and one line with a unique node 1
    # To sort chains, we need to start with the line with the unique node 0
    loops = []
    chains = []
    for edge in edges:
        starts, ends = tuple(zip(*edge))
        if set(starts) == set(ends):
            # To guarantee consistent behavior, arbitarily set the start node of a loop to the minimum node index
            loops.append({"start": min(starts), "edge": edge})
        else:
            chains.append({"start": list(set(starts) - set(ends))[0], "edge": edge})

    # Sort
    sorted_loops = [sort_edge(loop["edge"], loop["start"]) for loop in loops]
    sorted_chains = [sort_edge(chain["edge"], chain["start"]) for chain in chains]

    return sorted_loops, sorted_chains


def sort_edge(edge, start_node=None):
    """
    Sort an edge represented by a list of 2 Tuples

    Parameters
    ----------
    edge
    start_node

    Returns
    -------

    """
    sorted_edge = []
    edge = list(edge)

    if start_node is not None:
        start_index = np.argmax([line[0] == start_node for line in edge])
    else:
        start_index = 0

    sorted_edge.append(edge.pop(start_index))  # Start with first item
    for _ in range(len(edge)):
        # Next item in loop is index where the start of the line is the end of the current line
        next_index = np.argmax([line[0] == sorted_edge[-1][1] for line in edge])
        sorted_edge.append(edge.pop(next_index))

    return sorted_edge


def triangulate_loop_with_stitch(loop, points=None):
    """
    Triangulate a loop by stitching back and forth accross it.
    *Note* This algorithm can create self intersecting geometry in loops with concave sections

    Parameters
    ----------
    loop
        List of lines making up the loop to be triangulated. Lines are represented by list of two ints referring to
        indices in a points array
    points
        Array of points representing the 3D coordinates referred to by the elements of the loop.
        Unused for this algorithm

    Returns
    -------

    """
    loop = list(zip(*loop))[0]  # Just need to look at the line starts
    faces = [[loop[-1], loop[0], loop[1]]]
    next_up_node = 2  # Already used 2 nodes from start of loop, 1 from end
    next_down_node = -2
    for i in range(len(loop) - 3):
        # Next face always starts with the new node
        # If i is even, count up from 0, if i is odd, count down from -1
        if i % 2 == 0:
            new_node = loop[next_up_node]
            next_up_node += 1
            faces.append([new_node, faces[-1][0], faces[-1][2]])  # on even iterations, go to opposite node first
        else:
            new_node = loop[next_down_node]
            next_down_node -= 1
            faces.append([new_node, faces[-1][1], faces[-1][0]])  # on odd iterations, go to adjacent node first

    return faces


def triangulate_loop_with_nearest_neighbors(loop, points):
    """
    Triangulate loop by building triangles using the nearest neighbor point to existing triangle edges.
    Todo: ensure one side of each triangle is on the boundary (important for 3d boundaries)
    Parameters
    ----------
    loop
        List of lines making up the loop to be triangulated. Lines are represented by list of two ints referring to
        indices in a points array
    points
        Array of points representing the 3D coordinates referred to by the elements of the loop.
        Unused for this algorithm

    Returns
    -------

    """
    loop = list(zip(*loop))[0]  # Just need to look at where each line starts
    faces = []

    # Start with a single face consisting of point 0 and its nearest neighbors
    point_1 = loop[0]
    neighbors = sorted(loop, key=lambda neighbor: np.linalg.norm(points[point_1] - points[neighbor]))
    point_2 = neighbors[1]
    point_3 = neighbors[2]
    faces.append([point_3, point_2, point_1])

    # Recursively build faces off the first face
    continue_triangulating_with_nearest_neighbors(faces, loop, points)

    return faces


def continue_triangulating_with_nearest_neighbors(faces, loop, points):

    for a, b in itertools.combinations(faces[-1], 2):  # For each combination of points in a face
        # If the points are adjacent in the loop, they are on the edge and don't need to be built off
        points_adjacent = find_sequence(loop, [a, b], check_reverse=True) >= 0

        # If the line a,b is already found in two triangles, don't build any more
        line_in_two_faces = \
            np.count_nonzero([find_sequence(face, [a, b], check_reverse=True) >= 0 for face in faces]) >= 2

        if not points_adjacent and not line_in_two_faces:
            # If not, build another triangle with a, b and the nearest neighbor to a, and continue recursively
            # building triangles
            point_1 = a
            point_2 = b

            # Look for neighbors that are not a or b and don't already have a triangle with a
            searchloop = []
            for item in loop:
                if item not in [point_1, point_2]:
                    line_in_faces_a = [find_sequence(face, [a, item], check_reverse=True) >= 0 for face in faces]
                    line_in_faces_b = [find_sequence(face, [b, item], check_reverse=True) >= 0 for face in faces]

                    if not np.any(line_in_faces_a) and not np.any(line_in_faces_b):
                        searchloop.append(item)

            if not searchloop:
                continue

            neighbors = sorted(searchloop, key=lambda neighbor: np.linalg.norm(points[a] - points[neighbor]))
            point_3 = neighbors[0]
            faces.append([point_3, point_2, point_1])
            continue_triangulating_with_nearest_neighbors(faces, loop, points)



loop_triangulation_algorithms = {
    "stitch": triangulate_loop_with_stitch,
    "nearest_neighbor": triangulate_loop_with_nearest_neighbors
}


def select_intersecting_triangles(mesh: pv.PolyData, quiet=False, *args, **kwargs):
    """
    Wrapper around the pymeshfix function for selecting self intersecting triangles

    Parameters
    ----------
    mesh
    quiet
        Enable or disable verbose output from pymehsfix
        *NOTE* pymeshfix seems to have this backward. Quiet=True makes it loud. Quiet=False makes it quiet
    args
    kwargs

    Returns
    -------

    """
    tin = pymeshfix.PyTMesh(quiet)
    tin.load_array(mesh.points, pyvista_faces_to_2d(mesh.faces))
    intersecting = tin.select_intersecting_triangles(*args, **kwargs)
    return intersecting
