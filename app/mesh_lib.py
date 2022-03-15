from typing import Dict, Tuple, List, Optional

import gmsh
import numpy as np
import pymeshfix
import pymeshlab
import pymeshlab.pmeshlab
import pyvista as pv

from pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d


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


pymeshlab_op_map = {
    "Difference": pymeshlab.MeshSet.generate_boolean_difference.__name__,
    "Intersection": pymeshlab.MeshSet.generate_boolean_intersection.__name__,
    "Union": pymeshlab.MeshSet.generate_boolean_union.__name__,
    "Xor": pymeshlab.MeshSet.generate_boolean_xor.__name__
}


def pymeshlab_boolean(meshes: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], operation: str) \
        -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Run a pymesh boolean operation on two input meshes. The meshes are described as a platform agnostic Tuple of ndarrays
    representing mesh vertices and faces. The format is as follows:
        Tuple element 0: 3xN ndarray of float representing XYZ points in 3D space
        Tuple element 1: 3xN ndarray of int representing triangular faces composed of indices the points

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
        ml_mesh = pymeshlab.Mesh(mesh[0], mesh[1])
        ms.add_mesh(ml_mesh)

    func = getattr(ms, pymeshlab_op_map[operation])
    try:
        func(first_mesh=0, second_mesh=1)
    except pymeshlab.pmeshlab.PyMeshLabException:
        return None

    booleaned_mesh = (ms.mesh(2).vertex_matrix(), ms.mesh(2).face_matrix())

    return booleaned_mesh


def gmsh_load_from_arrays(mesh_vertices: np.ndarray, mesh_elements: np.ndarray, dim: int = 2, msh_type: int = 2):
    """
    Load a mesh into gmsh fram a set of vertex and face arrays.
    Gmsh must be initialized before using this function.

    Parameters
    ----------
    mesh
    dim
    msh_type
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