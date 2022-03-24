import gmsh
import numpy as np
from tetrahedralizer.mesh_lib import gmsh_load_from_arrays, gmsh_tetrahedral_mesh_to_arrays, gmsh_tetrahedralize, \
    preprocess_and_tetrahedralize, remove_shared_faces
import pyvista as pv
from tetrahedralizer.pyvista_tools import pyvista_faces_to_2d, pyvista_tetrahedral_mesh_from_arrays

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


def test_gmsh_tetrahedralize():
    outer = pv.Box(quads=False).scale([5, 5, 5], inplace=False)
    inner_a = pv.Box(quads=False).translate([2, 0, 0], inplace=False)
    inner_b = pv.Box(quads=False).translate([-2, 0, 0], inplace=False)

    mesh_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in [outer, inner_a, inner_b]]

    tet_nodes, tet_elements = gmsh_tetrahedralize(mesh_arrays, gmsh_options={})
    pv_tet = pyvista_tetrahedral_mesh_from_arrays(tet_nodes, tet_elements[1])

    # # Plot result
    # p = pv.Plotter()
    # p.add_mesh(pv_tet, opacity=0.15,  show_edges=True, edge_color="gray")
    #
    # def plane_func(normal, origin):
    #     slc = pv_tet.slice(normal=normal, origin=origin)
    #     p.add_mesh(slc, name="slice", show_edges=True)
    #
    # p.add_plane_widget(plane_func, assign_to_axis="z")
    # p.show()

    correct_points = np.array([
        [-5., -5., -5.], [5., -5., -5.], [-5., 5., -5.], [5., 5., -5.],
        [-5., -5., 5.], [5., -5., 5.], [-5., 5., 5.], [5., 5., 5.],
        [1., -1., -1.], [3., -1., -1.], [1., 1., -1.], [3., 1., -1.],
        [1., -1., 1.], [3., -1., 1.], [1., 1., 1.], [3., 1., 1.],
        [-3., -1., -1.], [-1., -1., -1.], [-3., 1., -1.], [-1., 1., -1.],
        [-3., -1., 1.], [-1., -1., 1.], [-3., 1., 1.], [-1., 1., 1.]])

    correct_cells = np.array([
        4, 5, 12, 21, 7, 4, 23, 4, 21, 7, 4, 3, 10, 19, 1, 4, 8,
        9, 5, 1, 4, 3, 9, 10, 1, 4, 21, 4, 0, 5, 4, 1, 2, 19,
        3, 4, 23, 4, 7, 6, 4, 4, 7, 5, 21, 4, 19, 8, 14, 17, 4,
        17, 14, 21, 12, 4, 19, 21, 14, 23, 4, 8, 19, 14, 10, 4, 17, 8,
        14, 12, 4, 21, 19, 14, 17, 4, 17, 18, 0, 16, 4, 7, 21, 23, 14,
        4, 9, 5, 13, 8, 4, 22, 16, 0, 18, 4, 0, 17, 16, 21, 4, 7,
        12, 21, 14, 4, 12, 13, 5, 8, 4, 15, 12, 7, 14, 4, 11, 7, 14,
        15, 4, 8, 19, 10, 1, 4, 9, 3, 10, 11, 4, 8, 10, 9, 1, 4,
        20, 23, 4, 21, 4, 5, 3, 1, 9, 4, 22, 19, 6, 23, 4, 3, 19,
        6, 2, 4, 21, 5, 17, 12, 4, 21, 17, 5, 0, 4, 8, 17, 5, 12,
        4, 1, 5, 17, 0, 4, 17, 5, 1, 8, 4, 1, 19, 17, 8, 4, 0,
        21, 20, 4, 4, 20, 21, 0, 16, 4, 0, 22, 20, 16, 4, 18, 0, 6,
        2, 4, 6, 0, 18, 22, 4, 6, 19, 18, 2, 4, 18, 19, 6, 22, 4,
        6, 23, 20, 22, 4, 20, 23, 6, 4, 4, 20, 0, 6, 22, 4, 6, 0,
        20, 4, 4, 13, 11, 5, 9, 4, 3, 11, 5, 7, 4, 3, 5, 11, 9,
        4, 12, 15, 5, 13, 4, 12, 5, 15, 7, 4, 11, 5, 15, 13, 4, 11,
        15, 5, 7, 4, 14, 7, 19, 23, 4, 6, 7, 19, 3, 4, 6, 19, 7,
        23, 4, 19, 10, 7, 14, 4, 19, 7, 10, 3, 4, 11, 7, 10, 14, 4,
        11, 10, 7, 3, 4, 1, 17, 2, 0, 4, 1, 2, 17, 19, 4, 18, 2,
        17, 0, 4, 18, 17, 2, 19])

    assert np.array_equal(pv_tet.points, correct_points)
    assert np.array_equal(pv_tet.cells, correct_cells)


def test_preprocess_and_tetrahedralize_simple():
    outer = pv.Box(quads=False).scale([5, 5, 5], inplace=False)
    inner_a = pv.Box(quads=False).translate([2, 0, 0], inplace=False)
    inner_b = pv.Box(quads=False).translate([-2, 0, 0], inplace=False)

    tet = preprocess_and_tetrahedralize(outer, [inner_a, inner_b], {}, {})

    # Plot result
    # p = pv.Plotter()
    # p.add_mesh(tet, opacity=0.15,  show_edges=True, edge_color="gray")
    #
    # def plane_func(normal, origin):
    #     slc = tet.slice(normal=normal, origin=origin)
    #     p.add_mesh(slc, name="slice", show_edges=True)
    #
    # p.add_plane_widget(plane_func, assign_to_axis="z")
    # p.show()

    correct_points = np.array([
        [-5., -5., -5.], [5., -5., -5.], [-5., 5., -5.], [5., 5., -5.], [-5., -5., 5.],
        [5., -5., 5.], [-5., 5., 5.], [5., 5., 5.], [1., -1., -1.], [3., -1., -1.],
        [1., 1., -1.], [3., 1., -1.], [1., -1., 1.], [3., -1., 1.], [1., 1., 1.],
        [3., 1., 1.], [-3., -1., -1.], [-1., -1., -1.], [-3., 1., -1.], [-1., 1., -1.],
        [-3., -1., 1.], [-1., -1., 1.], [-3., 1., 1.], [-1., 1., 1.], [1., -1., -1.],
        [3., -1., -1.], [1., 1., -1.], [3., 1., -1.], [1., -1., 1.], [3., -1., 1.],
        [1., 1., 1.], [3., 1., 1.], [2.07309185, -0.33974974, -0.33681687], [1.70284222, 0.2953999, 0.07849155],
        [-3., -1., -1.], [-1., -1., -1.], [-3., 1., -1.], [-1., 1., -1.], [-3., -1., 1.],
        [-1., -1., 1.], [-3., 1., 1.], [-1., 1., 1.], [-1.92690815, -0.33974974, -0.33681687],
        [-2.29715778, 0.2953999, 0.07849155]])

    correct_cells = np.array([4, 19, 3, 10, 1, 4, 8, 9, 5, 1, 4, 3, 9, 10, 1, 4, 4,
                              23, 6, 7, 4, 21, 4, 0, 5, 4, 19, 3, 1, 2, 4, 19, 8, 12,
                              17, 4, 14, 8, 19, 10, 4, 21, 19, 12, 17, 4, 23, 12, 19, 14, 4,
                              8, 14, 19, 12, 4, 19, 21, 12, 23, 4, 17, 18, 0, 16, 4, 20, 23,
                              4, 21, 4, 9, 5, 13, 8, 4, 22, 16, 0, 18, 4, 0, 17, 16, 21,
                              4, 12, 13, 5, 8, 4, 11, 7, 14, 15, 4, 3, 5, 9, 1, 4, 8,
                              19, 10, 1, 4, 9, 3, 10, 11, 4, 8, 10, 9, 1, 4, 12, 5, 23,
                              21, 4, 4, 5, 23, 7, 4, 4, 23, 5, 21, 4, 22, 19, 6, 23, 4,
                              3, 19, 6, 2, 4, 21, 5, 17, 12, 4, 21, 17, 5, 0, 4, 8, 17,
                              5, 12, 4, 1, 5, 17, 0, 4, 17, 5, 1, 8, 4, 1, 19, 17, 8,
                              4, 0, 21, 20, 4, 4, 20, 21, 0, 16, 4, 0, 22, 20, 16, 4, 18,
                              0, 6, 2, 4, 6, 0, 18, 22, 4, 6, 19, 18, 2, 4, 18, 19, 6,
                              22, 4, 1, 17, 2, 0, 4, 1, 2, 17, 19, 4, 18, 2, 17, 0, 4,
                              18, 17, 2, 19, 4, 6, 23, 20, 22, 4, 20, 23, 6, 4, 4, 20, 0,
                              6, 22, 4, 6, 0, 20, 4, 4, 13, 11, 5, 9, 4, 3, 11, 5, 7,
                              4, 3, 5, 11, 9, 4, 6, 19, 7, 23, 4, 6, 7, 19, 3, 4, 14,
                              7, 19, 23, 4, 12, 15, 5, 13, 4, 11, 5, 15, 13, 4, 11, 15, 5,
                              7, 4, 19, 10, 7, 14, 4, 19, 7, 10, 3, 4, 11, 7, 10, 14, 4,
                              11, 10, 7, 3, 4, 14, 15, 5, 12, 4, 5, 15, 14, 7, 4, 5, 23,
                              14, 12, 4, 14, 23, 5, 7, 4, 28, 29, 24, 32, 4, 29, 25, 24, 32,
                              4, 25, 26, 24, 32, 4, 27, 26, 25, 32, 4, 27, 26, 32, 33, 4, 26,
                              27, 30, 33, 4, 29, 27, 25, 32, 4, 26, 24, 32, 33, 4, 24, 28, 32,
                              33, 4, 26, 30, 24, 33, 4, 30, 28, 24, 33, 4, 33, 30, 31, 27, 4,
                              33, 31, 30, 28, 4, 29, 32, 31, 27, 4, 29, 31, 32, 28, 4, 32, 33,
                              31, 27, 4, 32, 31, 33, 28, 4, 38, 39, 34, 42, 4, 39, 35, 34, 42,
                              4, 35, 36, 34, 42, 4, 37, 36, 35, 42, 4, 37, 36, 42, 43, 4, 36,
                              37, 40, 43, 4, 39, 37, 35, 42, 4, 36, 34, 42, 43, 4, 34, 38, 42,
                              43, 4, 36, 40, 34, 43, 4, 40, 38, 34, 43, 4, 43, 40, 41, 37, 4,
                              43, 41, 40, 38, 4, 39, 42, 41, 37, 4, 39, 41, 42, 38, 4, 42, 43,
                              41, 37, 4, 42, 41, 43, 38])

    assert np.all(np.isclose(tet.points, correct_points))
    assert np.array_equal(tet.cells, correct_cells)




