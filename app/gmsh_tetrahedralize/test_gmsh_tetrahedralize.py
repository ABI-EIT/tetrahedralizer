import gmsh
import numpy as np
from gmsh_tetrahedralize import gmsh_load_from_arrays, gmsh_tetrahedral_mesh_to_arrays

mesh_a_verts = np.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0],
                         [1, 1, 1],
                         [0, 0.5, 0.5],
                         [1, 0.5, 0.5]], dtype=float)

mesh_a_faces = np.array([[0, 1, 5], [0, 5, 4],
                         [1, 3, 7], [1, 7, 5],
                         [3, 2, 6], [3, 6, 7],
                         [2, 0, 4], [2, 4, 6],
                         [4, 5, 9], [5, 7, 9], [7, 6, 9], [6, 4, 9],
                         [0, 1, 8], [1, 3, 8], [3, 2, 8], [2, 0, 8]])

mesh_a_default_tets_gmsh = np.array([[5, 8, 7, 9],
                                     [7, 9, 8, 6],
                                     [9, 5, 8, 4],
                                     [4, 6, 8, 9],
                                     [4, 2, 8, 6],
                                     [8, 1, 7, 3],
                                     [0, 5, 8, 1],
                                     [8, 5, 7, 1],
                                     [2, 3, 8, 6],
                                     [5, 0, 8, 4],
                                     [2, 4, 8, 0],
                                     [7, 8, 3, 6]], dtype=np.uint64)


def test_gmsh_load_from_arrays():
    gmsh.initialize()
    gmsh_load_from_arrays(mesh_a_verts, mesh_a_faces)

    _, coord, _ = gmsh.model.mesh.getNodes()
    nodes_array = np.array(coord).reshape(-1, 3)
    element_types, _, node_tags = gmsh.model.mesh.getElements()
    elements_array = (np.array(node_tags) - 1).reshape(-1, 3)
    gmsh.finalize()

    assert np.array_equal(nodes_array, mesh_a_verts)
    assert np.array_equal(elements_array, mesh_a_faces)
    assert np.array_equal(element_types, [2])


def test_gmsh_tetrahedral_mesh_to_arrays():
    # load from arrays
    gmsh.initialize()
    gmsh_load_from_arrays(mesh_a_verts, mesh_a_faces)

    # tetrahedralize
    s = gmsh.model.get_entities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    nodes, (tris, tets) = gmsh_tetrahedral_mesh_to_arrays()

    mesh_a_faces.sort(axis=0)
    tris.sort(axis=0)

    assert np.array_equal(nodes, mesh_a_verts)
    assert np.array_equal(tris, mesh_a_faces)
    assert np.array_equal(tets, mesh_a_default_tets_gmsh)

    gmsh.finalize()
