import itertools
from typing import Dict, Tuple, List, Optional
import gmsh
import numpy as np
import pymeshfix
from pymeshlab.pmeshlab import PyMeshLabException
import pymeshlab
import pyvista as pv
from tetrahedralizer import pyvista_tools
from tetrahedralizer.pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d, remove_shared_faces_with_ray_trace,\
    remove_shared_faces


def fix_mesh(mesh: pv.DataSet, repair_kwargs: Dict = None) -> Tuple[pv.DataSet, pv.PolyData]:
    """
    Call the meshfix.repair function on a Pyvista dataset and return a fixed copy of the dataset along with the meshfix
    holes

    Parameters
    ----------
    mesh
        Pyvista Dataset
    repair_kwargs
        Kwargs for meshfix.repair

    Returns
    -------
        Fixed copy of the input mesh, holes identified by meshfix

    """

    if repair_kwargs is None:
        repair_kwargs = {}
    meshfix = pymeshfix.MeshFix(mesh.points, pyvista_faces_to_2d(mesh.faces))
    holes = meshfix.extract_holes()
    meshfix.repair(**repair_kwargs)
    fixed_mesh = mesh.copy()
    fixed_mesh.points = meshfix.v
    fixed_mesh.faces = pyvista_faces_to_1d(meshfix.f)

    return fixed_mesh, holes


#: Dict to map names to pymeshlab boolean operations
pymeshlab_op_map = {
    "Difference": pymeshlab.MeshSet.generate_boolean_difference.__name__,
    "Intersection": pymeshlab.MeshSet.generate_boolean_intersection.__name__,
    "Union": pymeshlab.MeshSet.generate_boolean_union.__name__,
    "Xor": pymeshlab.MeshSet.generate_boolean_xor.__name__
}


def pymeshlab_boolean(meshes: Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]], operation: str) \
        -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Run a pymesh boolean operation on two input meshes. The meshes are described as a platform agnostic Tuple of ndarrays
    representing mesh vertices and faces. The format is as follows:
    Tuple element 0: 3xN ndarray of float representing XYZ points in 3D space
    Tuple element 1: 3xN ndarray of int representing triangular faces composed of indices the points

    Optionally, face normals can be specified:
    Tuple element 2: 3xN ndarray of float representing face normals. But this doesn't work

    Parameters
    ----------
    meshes
        Tuple of two input meshes in array format
    operation
        Pymeshlab boolean operation name. String names are mapped to pymeshlab functions in ppymeshlab_op_map

    Returns
    -------
    booleaned_mesh
        Result of the boolean operation in array format

    """

    ms = pymeshlab.MeshSet()
    for mesh in meshes:
        if len(mesh) == 2:
            ml_mesh = pymeshlab.Mesh(mesh[0], mesh[1])
        elif len(mesh) == 3:
            ml_mesh = pymeshlab.Mesh(mesh[0], mesh[1], f_normals_matrix=mesh[2])
        else:
            raise ValueError("Incorrect number of elements in mesh Tuple")
        ms.add_mesh(ml_mesh)

    func = getattr(ms, pymeshlab_op_map[operation])
    try:
        func(first_mesh=0, second_mesh=1)
    except PyMeshLabException:
        return None

    booleaned_mesh = (ms.mesh(2).vertex_matrix(), ms.mesh(2).face_matrix())

    return booleaned_mesh


def gmsh_load_from_arrays(mesh_vertices: np.ndarray, mesh_elements: np.ndarray, dim: int = 2, msh_type: int = 2):
    """
    Load a mesh into gmsh fram a set of vertex and face arrays. Gmsh must be initialized before using this function.

    Parameters
    ----------
    mesh_vertices
        3xN ndarray of float representing XYZ points in 3D space
    mesh_elements
        3xN ndarray of int representing triangular faces composed of indices the points
    dim
        gmsh mesh dimension
    msh_type
        gmsh mesh type
    """
    mesh = (mesh_vertices, mesh_elements)
    tag = gmsh.model.add_discrete_entity(dim)

    max_node = gmsh.model.mesh.getMaxNodeTag()
    node_tags = max_node + 1 + np.array(range(mesh[0].shape[0]))
    gmsh.model.mesh.add_nodes(dim, tag, node_tags, mesh[0].ravel())

    max_element = gmsh.model.mesh.getMaxElementTag()
    element_tags = max_element + np.array(range(mesh[1].shape[0]))
    element_node_tags = mesh[1] + node_tags[0]
    gmsh.model.mesh.add_elements(dim, tag, [msh_type], [element_tags], [element_node_tags.ravel()])


def gmsh_tetrahedral_mesh_to_arrays() -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Obtain an array representation of the tetrahedral mesh in gmsh, if it exists.

    Returns
    -------
    nodes_array, (triangles, tets)
        3xN array of points, (3xN array of triangular faces, 4xN array of tetrahedral cells)

    """
    _, coord, _ = gmsh.model.mesh.getNodes()
    nodes_array = np.array(coord).reshape(-1, 3)
    element_types, _, node_tags = gmsh.model.mesh.getElements()
    node_tags = [np.array(tag_set) - 1 for tag_set in node_tags]
    if 2 in element_types:
        triangles = node_tags[np.argmax(element_types == 2)].reshape(-1, 3)
    else:
        raise NotFoundError("Surface not found in gmsh")
    if 4 in element_types:
        tets = node_tags[np.argmax(element_types == 4)].reshape(-1, 4)
    else:
        raise NotFoundError("Tetrahedra not found in gmsh")

    return nodes_array, (triangles, tets)


class NotFoundError(Exception):
    pass


def gmsh_tetrahedralize(meshes: List[Tuple[np.ndarray, np.ndarray]], gmsh_options: dict) \
        -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Run the gmsh tetrahedralization operation (mesh.generate(3)) on a group of surface meshes. Gmsh will interpret the
    outermost mesh as the outer surface, and all other meshes as holes to generte in the tetrahedralization.

    The meshes are described as a platform agnostic Tuple of ndarrays
    representing mesh vertices and faces. The format is as follows:
    Tuple element 0: 3xN ndarray of float representing XYZ points in 3D space
    Tuple element 1: 3xN ndarray of int representing triangular faces composed of indices the points


    Parameters
    ----------
    meshes
        List of meshes to tetrahedralize
    gmsh_options
        Dict of values to be passed to gmsh.option.set_number
    Returns
    -------
        Tetrahedralized mesh. Elements are now represented as a Tuple of ndarray. Element 0 is a 3xN array of surface
        faces. Element 1 is a 4xN array of tetrahedral cells

    """
    gmsh.initialize()
    for mesh in meshes:
        gmsh_load_from_arrays(mesh[0], mesh[1])

    # Create a volume from all the surfaces
    s = gmsh.model.get_entities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])

    gmsh.model.geo.synchronize()

    for name, value in gmsh_options.items():
        gmsh.option.set_number(name, value)

    gmsh.model.mesh.generate(3)

    nodes, elements = gmsh_tetrahedral_mesh_to_arrays()
    gmsh.finalize()
    return nodes, elements


def preprocess_and_tetrahedralize(outer_mesh: pv.PolyData, inner_meshes: List[pv.PolyData], mesh_repair_kwargs: dict,
                                  gmsh_options: dict) -> pv.UnstructuredGrid:
    """
    Automatically create a tetrahedralization from multiple input surface meshes. The outer mesh represents the
    outermost boundary of the output mesh, and the inner meshes represent the boundaries of individual inner sections
    of the output mesh. The process is configurable with mesh_repair_kwargs and gmsh_options.

    Parameters
    ----------
    outer_mesh
        Pyvista DataSet representing the outer surface
    inner_meshes
        List of Pyvista DataSets representing the surfaces of inner sections
    mesh_repair_kwargs
        Kwargs for PyMeshfix
    gmsh_options
        Kwargs for gmsh

    Returns
    -------
    combined
        Pyvista unstructured grid

    """
    print("Fixing...")
    # Fix all inputs
    fixed_meshes = []
    for mesh in [outer_mesh, *inner_meshes]:
        if not mesh.is_manifold:
            fixed_meshes.append(fix_mesh(mesh, mesh_repair_kwargs)[0])
        else:
            fixed_meshes.append(mesh)

    fixed_mesh_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in fixed_meshes]

    print("Diffing...")
    # Check all pairs of inner meshes for intersections and create:
    # # List of meshes where intersecting pairs are replaced with a diffed and an original
    diffed_meshes = dif_any_intersecting(fixed_mesh_arrays[1:])
    pv_diffed_meshes = [pv.PolyData(mesh[0], pyvista_faces_to_1d(mesh[1])) for mesh in diffed_meshes]
    fixed_diffed = [fix_mesh(mesh, mesh_repair_kwargs)[0] if not mesh.is_manifold else mesh for mesh in pv_diffed_meshes]

    print("Combining...")
    # Remove shared faces to form inner hole
    combined = remove_shared_faces(inner_meshes, progress_bar=True)
    fixed_combined = [fix_mesh(mesh)[0] if not mesh.is_manifold else mesh for mesh in combined]
    fixed_combined_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in fixed_combined]

    print("Unioning...")
    # Check all pairs of inner meshes for intersections and create:
    # # List of meshes where intersecting sets are replaced with a union
    unioned_meshes = union_any_intersecting(fixed_combined_arrays)
    pv_unioned_meshes = [pv.PolyData(mesh[0], pyvista_faces_to_1d(mesh[1])) for mesh in unioned_meshes]
    fixed_unioned = [fix_mesh(mesh, mesh_repair_kwargs)[0] if not mesh.is_manifold else mesh for mesh in pv_unioned_meshes]

    print("Tetrahedralizing...")
    fixed_unioned_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in fixed_unioned]
    # Tetrahedralize outer mesh with hole, then convert to pyvista
    nodes, elements = gmsh_tetrahedralize([fixed_mesh_arrays[0], *fixed_unioned_arrays], gmsh_options)
    outer_tetrahedralized = pyvista_tools.pyvista_tetrahedral_mesh_from_arrays(nodes, elements[1])

    # Tetrahedralize each inner mesh, then convert to pyvista
    inner_tetrahedralized = []
    fixed_diffed_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in fixed_diffed]
    for mesh in fixed_diffed_arrays:
        nodes, elements = gmsh_tetrahedralize([mesh], gmsh_options)
        inner_tetrahedralized.append(
            pyvista_tools.pyvista_tetrahedral_mesh_from_arrays(nodes, elements[1]))

    # Combine result
    out_meshes = [outer_tetrahedralized, *inner_tetrahedralized]
    for i, mesh in enumerate(out_meshes):
        mesh.cell_data["Scalar"] = np.asarray([i % len(out_meshes)] * mesh.n_cells)
    blocks = pv.MultiBlock(out_meshes)
    out_combined = blocks.combine()

    cell_sizes = out_combined.compute_cell_sizes()

    outer_surface_volume = fixed_meshes[0].volume
    total_cell_volume = sum(cell_sizes["Volume"])
    mean_cell_volume = np.mean(cell_sizes["Volume"])
    hole_volume = outer_surface_volume - total_cell_volume
    hole_volume_percent = (hole_volume/outer_surface_volume)*100
    hole_volume_in_cells = hole_volume/mean_cell_volume

    print(f"Hole volume: {hole_volume_percent:.4f}% of outer surface, {hole_volume_in_cells:.4f} x mean cell volume")

    return out_combined


def union_any_intersecting(meshes: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Iterate through a list of surfaces and perform a boolean union on any that are intersecting.

    Parameters
    ----------
    meshes
        Input meshes represented as a Tuple of ndarray.
            Tuple element 0: 3xN ndarray of float representing XYZ points in 3D space
            Tuple element 1: 3xN ndarray of int representing triangular faces composed of indices the points

    Returns
    -------
    unioned_meshes
        Processed input meshes where intersecting meshes are unioned, and non intersecting meshes are returned
        unchanged.

    """
    intersection_sets = []

    # Iterate through all pairs and create sets of intersecting meshes
    for (index_a, mesh_a), (index_b, mesh_b) in itertools.combinations(enumerate(meshes), 2):
        # If they intersect
        if pymeshlab_boolean((mesh_a, mesh_b), operation="Intersection") is not None:
            # If a is already part of a set, add b to it
            if np.any([index_a in s for s in intersection_sets]):
                intersection_sets[np.argmax([index_a in s for s in intersection_sets])].add(index_b)
            # Else if b is already part of a set, add a to it
            elif np.any([index_b in s for s in intersection_sets]):
                intersection_sets[np.argmax([index_b in s for s in intersection_sets])].add(index_a)
            # Else make a new one with both
            else:
                intersection_sets.append(set([index_a, index_b]))

    # Cumulatively union all meshes in each set
    unioned_meshes = []
    for intersection_set in intersection_sets:
        set_list = list(intersection_set)
        union_result = pymeshlab_boolean((meshes[set_list[0]], meshes[set_list[1]]), operation="Union")
        for index in set_list[2:]:
            union_result = pymeshlab_boolean((union_result, meshes[index]), operation="Union")
        unioned_meshes.append(union_result)

    # Put back in any that weren't unioned
    unioned_indices = list(itertools.chain(*intersection_sets))
    for index, mesh in enumerate(meshes):
        if index not in unioned_indices:
            unioned_meshes.append(mesh)

    return unioned_meshes


def dif_any_intersecting(meshes: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Iterate through a list of input surfaces and perform a boolean difference on any that are intersercting.

    Parameters
    ----------
    meshes
        Input meshes represented as a Tuple of ndarray.
            Tuple element 0: 3xN ndarray of float representing XYZ points in 3D space
            Tuple element 1: 3xN ndarray of int representing triangular faces composed of indices the points

    Returns
    -------
    diffed_meshes
        Processed input meshes where intersecting meshes are diffed, and non intersecting meshes are returned
        unchanged.

    """
    intersection_list = []
    dif_pairs = []

    # Iterate through all pairs and create sets of intersecting meshes
    index_b: object
    for (index_a, mesh_a), (index_b, mesh_b) in itertools.combinations(enumerate(meshes), 2):
        # If they intersect
        if pymeshlab_boolean((mesh_a, mesh_b), operation="Intersection") is not None:
            # If we've already seen a, add b dif a to dif pairs
            if index_a in intersection_list:
                intersection_list.append(index_b)
                dif_pairs.append((index_b, index_a))
            # Else if we've already seen b, add a dif b to dif pairs
            elif index_b in intersection_list:
                intersection_list.append(index_a)
                dif_pairs.append((index_a, index_b))
            # Else we've now seen both and add a dif b to dif pairs
            else:
                intersection_list.extend([index_a, index_b])
                dif_pairs.append((index_a, index_b))

    # Diff all the pairs
    diffed_meshes = []
    for pair in dif_pairs:
        diffed_meshes.append(pymeshlab_boolean((meshes[pair[0]], meshes[pair[1]]), operation="Difference"))

    # Put back in any that weren't diffed
    diffed_indices = [pair[0] for pair in dif_pairs]
    for index, mesh in enumerate(meshes):
        if index not in diffed_indices:
            diffed_meshes.append(mesh)

    return diffed_meshes



