from __future__ import annotations
import itertools
from typing import List, Tuple, Optional
import numpy as np
import pyvista
import vtkmodules.util
import pyvista as pv
from tqdm import tqdm
from tetrahedralizer.pyvista_tools.geometry_tools import find_loops_and_chains, dihedral_angle, find_sequence
from tetrahedralizer.pyvista_tools import select_faces_using_points, select_shared_points, pyvista_faces_to_1d, \
    compute_face_agreement_with_normals, rewind_face, identify_neighbors, pyvista_faces_to_2d, choose_surface_face, \
    loop_triangulation_algorithms, compute_neighbor_angles

"""
pyvista_features is a module that provides high level features for working with pyvista objects. These features should
transform or create pyvista objects. They are intended to be useful on their own as part of a workflow that makes use
of pyvista objects.
"""


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
                    (dihedral_angle(mesh_a.cell_normals[intersection_cell], mesh_b.cell_normals[i]) % (np.pi / 2))[0]
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
    output, (faces_to_remove)
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


def rewind_faces_to_normals(mesh: pv.PolyData, inplace=False) -> Optional[pv.PolyData]:
    """
    Re-order the faces of a Pyvista Polydata to match the face normals

    Parameters
    ----------
    mesh
        Mesh to reorder the faces of
    inplace
        Update mesh inplace

    Returns
    -------
    mesh_c
        Updated mesh

    """
    mesh_c = mesh.copy()

    agreements = compute_face_agreement_with_normals(mesh_c)
    for i in range(mesh.n_faces):
        if not agreements[i]:
            mesh_c = rewind_face(mesh_c, i)

    if inplace:
        mesh.faces = mesh_c.faces
    else:
        return mesh_c


def extract_outer_surface(surface: pv.PolyData, inplace=False) -> Optional[pv.PolyData]:
    """
    An algorithm to refine a surface mesh by keeping only faces on the outer surface of the mesh, thereby removing
    any inner walls that would be left by the Pyvista extract surface algorithm.

    This algorithm starts by identifying a known surface face, then recursively adds connected faces which lie on the
    surface. This is necessary instead of iterating through each face because the method for determining a surface face
    on a non-manifold line requires knowledge of another surface face.

    Parameters
    ----------
    surface
        Surface to refine
    inplace
        Update mesh inplace

    Returns
    -------
    r_surface
        Refined surface

    """
    r_surface = surface.copy()

    # Find a face on the outer surface by casting a long ray from the first surface and choosing the last face it hits
    stop = r_surface.cell_centers().points[0] - r_surface.face_normals[0]
    b = r_surface.bounds
    distance = np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    start = stop + (r_surface.face_normals[0] * distance)
    _, intersection_cells = r_surface.ray_trace(start, stop, first_point=True)
    face_a = intersection_cells[0]

    # p = pv.Plotter()
    # line = pv.Line(start, stop)
    # p.add_mesh(surface, style="wireframe")
    # p.add_mesh(line)
    # p.show()

    neighbors_dict, _ = identify_neighbors(r_surface)

    # Recursively select faces which belong to the true surface
    selected_faces = []
    continue_extracting_outer_surface(r_surface, selected_faces, face_a, neighbors_dict)
    r_surface.faces = pyvista_faces_to_1d(pyvista_faces_to_2d(r_surface.faces)[selected_faces])
    r_surface = r_surface.clean()

    if inplace:
        surface.overwrite(r_surface)
    else:
        return r_surface


def continue_extracting_outer_surface(surface: pv.PolyData, selected_faces: list, face: int, neighbors_dict: dict):
    """
    Recursively move through neighbors of a face, selecting which neighbors belong to the true surface of the given
    surface mesh.

    Parameters
    ----------
    surface
    selected_faces
    face
    neighbors_dict
    """
    for line in neighbors_dict[face]:
        neighbor_list = neighbors_dict[face][line]
        if len(neighbor_list) > 1:
            if not np.any([neighbor in selected_faces for neighbor in neighbor_list]):
                chosen_neighbor = choose_surface_face(surface, face, neighbor_list, line)
                selected_faces.append(chosen_neighbor)
                continue_extracting_outer_surface(surface, selected_faces, chosen_neighbor, neighbors_dict)
        else:
            neighbor = neighbor_list[0]
            if neighbor not in selected_faces:
                selected_faces.append(neighbor)
                continue_extracting_outer_surface(surface, selected_faces, neighbor, neighbors_dict)


def repeatedly_fill_holes(mesh: pv.DataSet, max_iterations=10, inplace=False, hole_size=1000, **kwargs) \
        -> Optional[pv.DataSet]:
    """
    Repeatedly run the pyvista fill holes function on a dataset.

    Parameters
    ----------
    mesh
        Mesh to fill holes in
    max_iterations
        Maximum number of times to fill holes
    inplace
        Update the mesh inplace
    hole_size
        Hole size argument for pv.DataSet.fill_holes
    kwargs
        kwargs for pv.Dataset.fill_holes

    Returns
    -------
    out
        Mesh with holes filled

    """
    out = mesh.copy()
    for _ in range(max_iterations):
        out = out.fill_holes(hole_size=hole_size, **kwargs)
        if out.is_manifold:
            break

    if inplace:
        mesh.overwrite(out)
    else:
        return out


def fill_holes_with_strategy(mesh: pv.PolyData, strategy: str | callable = "stitch", inplace=False) -> Optional[pv.PolyData]:
    """
    Fill holes in a Pyvista PolyData mesh using a specified algorithm.

    Todo: add max hole size

    Parameters
    ----------
    mesh
        Mesh to fill holes in
    strategy
        Hole filling strategy. Can be a string referring to an algorithm in pyvista_tools.loop_triangulation_algorithms,
        or a callable implementing the interface of the loop_triangulation_algorithms
    inplace
        Update mesh inplace

    Returns
    -------
    fill_mesh
        Mesh with holes filled

    """
    if isinstance(strategy, str):
        loop_triangulation_strategy = loop_triangulation_algorithms[strategy]
    else:
        loop_triangulation_strategy = strategy

    fill_mesh = mesh.copy()
    # Extract boundary edges
    boundaries = fill_mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                                 feature_edges=False, manifold_edges=False)

    # Find loops
    loops, _ = find_loops_and_chains(pyvista_faces_to_2d(boundaries.lines))
    # Triangulate
    patches = [loop_triangulation_strategy(loop, boundaries.points) for loop in loops]

    patches_surface = pv.PolyData(boundaries.points, pyvista_faces_to_1d(np.array(list(itertools.chain(*patches)))))

    # p = pv.Plotter()
    # p.add_mesh(patches_surface, style="wireframe")
    # loops_mesh = pv.PolyData(boundaries.points, lines=pyvista_faces_to_1d(list(itertools.chain(*loops))))
    # p.add_mesh(loops_mesh, color="red")
    # p.show()

    fill_mesh = fill_mesh.merge(patches_surface)

    if inplace:
        mesh.overwrite(fill_mesh)
    else:
        return fill_mesh


def remove_boundary_faces_recursively(mesh: pv.PolyData, inplace=False) -> Optional[pv.PolyData]:
    """
    Remove boundary faces, then re-check for boundary faces and continue removing recursively until no boundary faces
    remain.

    Parameters
    ----------
    mesh
    inplace

    Returns
    -------
    r_mesh
        Mesh with boundary faces removed

    """
    r_mesh = mesh.copy()

    neighbors_dict, lines_dict = identify_neighbors(mesh)

    boundary_faces = set()
    continue_selecting_boundary_faces(neighbors_dict, lines_dict, boundary_faces)

    r_mesh = r_mesh.remove_cells(list(boundary_faces))

    if inplace:
        mesh.overwrite(r_mesh)
    else:
        return r_mesh


def continue_selecting_boundary_faces(neighbors_dict, lines_dict, boundary_faces):
    # Find faces that are sole users of a line
    new_boundary_faces = set()
    for _, faces in lines_dict.items():
        if len(faces) == 1:
            new_boundary_faces.add(faces[0])

    if new_boundary_faces:
        # Find lines that are used by boundary faces
        check_lines = {face: list(lines.keys()) for face, lines in neighbors_dict.items() if face in new_boundary_faces}

        # Remove these boundary faces from all lines that use them.
        for face, lines in check_lines.items():
            for line in lines:
                lines_dict[line].remove(face)

        boundary_faces.update(new_boundary_faces)

        continue_selecting_boundary_faces(neighbors_dict, lines_dict, boundary_faces)


def extract_enclosed_regions(mesh: pv.PolyData) -> List[pv.PolyData]:
    """
    Extract enclosed regions from a surface mesh.

    Parameters
    ----------
    mesh
    """
    mesh = remove_boundary_faces_recursively(mesh)
    neighbors_dict, _ = identify_neighbors(mesh)

    region_sets = [set()]
    branch_points = []
    face = 0

    continue_extracting_enclosed_regions(mesh, face, region_sets, branch_points, neighbors_dict)

    regions = []
    for region_set in region_sets:
        faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh.face)[region_set])
        region = pv.PolyData(mesh.points, faces)
        region = region.clean()  # Remove unused points
        regions.append(region)

    return regions


def continue_extracting_enclosed_regions(mesh, face, region_sets, branch_faces, neighbors_dict):
    # TODO: the problem is that with inner faces, their normal could be in any direction. so we can accidentally
    # Jump back to the wrong region
    # Solution: always recalculate the normal for the next face to be "opposing" the one from the current face.
    # (i.e. both facing outward, the faces would rotate and hit each other instead of chasing each other

    # Add face to the current region set
    region_sets[-1].add(face)

    # Check all neighbors of the face
    for line, neighbors in neighbors_dict[face].items():
        # Update winding for all neighbors. Note: we could also do this after selecting the next face, but then we'd
        # have to do it when branching too, and we'd have to remember which neighbor we got to the branch from.
        for neighbor in neighbors:
            if find_sequence(pyvista_faces_to_2d(mesh.faces)[neighbor], line, check_reverse=False) == -1:
                rewind_face(mesh, neighbor)

        # If there are multiple neighbors sharing a line, choose the inner path, and add the rest to the branch list
        # so we can go back for them once this set is finished
        if len(neighbors) > 1:
            neighbors_angles = compute_neighbor_angles(mesh, face, neighbors, line, use_winding_order_normal=True)
            sorted_neighbors = [neighbor for _, neighbor in sorted(zip(neighbors_angles, neighbors), reverse=True)]
            # Neighbor with max angle (with positive defined as outward) is the inner path
            next_face = sorted_neighbors[0]
            for neighbor in sorted_neighbors[1:]:
                if neighbor not in branch_faces:
                    branch_faces.append(neighbor)
        else:
            next_face = neighbors[0]

        # Only continue if the next face is not already in the set
        if next_face not in region_sets[-1]:
            continue_extracting_enclosed_regions(mesh, next_face, region_sets,
                                                 branch_faces, neighbors_dict)

    # Once we finish a set, if we've assigned all faces, we're finished.
    # If not, we start a new set with the earliest branch
    all_faces = set(range(0, mesh.n_faces))
    assigned_faces = set().union(*region_sets)
    if not all_faces.issubset(assigned_faces):
        region_sets.append(set())
        branch_face = branch_faces.pop(0)
        continue_extracting_enclosed_regions(mesh, branch_face, region_sets, branch_faces, neighbors_dict)
