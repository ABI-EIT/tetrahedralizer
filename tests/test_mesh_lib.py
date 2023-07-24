import gmsh
import numpy as np
from tetrahedralizer.mesh_lib import gmsh_load_from_arrays, gmsh_tetrahedral_mesh_to_arrays, gmsh_tetrahedralize, \
    preprocess_and_tetrahedralize, pymeshlab_boolean, dif_any_intersecting
import pyvista as pv
from pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d, \
    pyvista_tetrahedral_mesh_from_arrays, rewind_faces_to_normals
import numpy.testing
from matplotlib import cm

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

    pv_tet = gmsh_tetrahedralize([outer, inner_a, inner_b], gmsh_options={})

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

    # # Plot result
    # p = pv.Plotter()
    # p.add_mesh(tet, opacity=0.15,  show_edges=True, edge_color="gray")
    #
    # def plane_func(normal, origin):
    #     slc = tet.slice(normal=normal, origin=origin)
    #     p.add_mesh(slc, name="slice", show_edges=True)
    #
    # p.add_plane_widget(plane_func, assign_to_axis="z")
    # p.show()

    correct_points = np.array([[-5.        , -5.        , -5.        ],
                 [ 5.        , -5.        , -5.        ],
                 [-5.        ,  5.        , -5.        ],
                 [ 5.        ,  5.        , -5.        ],
                 [-5.        , -5.        ,  5.        ],
                 [ 5.        , -5.        ,  5.        ],
                 [-5.        ,  5.        ,  5.        ],
                 [ 5.        ,  5.        ,  5.        ],
                 [ 1.        , -1.        , -1.        ],
                 [ 3.        , -1.        , -1.        ],
                 [ 1.        ,  1.        , -1.        ],
                 [ 3.        ,  1.        , -1.        ],
                 [ 1.        , -1.        ,  1.        ],
                 [ 3.        , -1.        ,  1.        ],
                 [ 1.        ,  1.        ,  1.        ],
                 [ 3.        ,  1.        ,  1.        ],
                 [-3.        , -1.        , -1.        ],
                 [-1.        , -1.        , -1.        ],
                 [-3.        ,  1.        , -1.        ],
                 [-1.        ,  1.        , -1.        ],
                 [-3.        , -1.        ,  1.        ],
                 [-1.        , -1.        ,  1.        ],
                 [-3.        ,  1.        ,  1.        ],
                 [-1.        ,  1.        ,  1.        ],
                 [ 2.07309185, -0.33974974, -0.33681687],
                 [ 1.70284222,  0.2953999 ,  0.07849155],
                 [-1.92690815, -0.33974974, -0.33681687],
                 [-2.29715778,  0.2953999 ,  0.07849155]])

    correct_cells = np.array([ 4,  5, 12, 21,  7,  4, 23,  4, 21,  7,  4,  3, 10, 19,  1,  4,  8,
        9,  5,  1,  4,  3,  9, 10,  1,  4, 21,  4,  0,  5,  4,  1,  2, 19,
        3,  4, 23,  4,  7,  6,  4,  4,  7,  5, 21,  4, 19,  8, 14, 17,  4,
       17, 14, 21, 12,  4, 19, 21, 14, 23,  4,  8, 19, 14, 10,  4, 17,  8,
       14, 12,  4, 21, 19, 14, 17,  4, 17, 18,  0, 16,  4,  7, 21, 23, 14,
        4,  9,  5, 13,  8,  4, 22, 16,  0, 18,  4,  0, 17, 16, 21,  4,  7,
       12, 21, 14,  4, 12, 13,  5,  8,  4, 15, 12,  7, 14,  4, 11,  7, 14,
       15,  4,  8, 19, 10,  1,  4,  9,  3, 10, 11,  4,  8, 10,  9,  1,  4,
       20, 23,  4, 21,  4,  5,  3,  1,  9,  4, 22, 19,  6, 23,  4,  3, 19,
        6,  2,  4, 21,  5, 17, 12,  4, 21, 17,  5,  0,  4,  8, 17,  5, 12,
        4,  1,  5, 17,  0,  4, 17,  5,  1,  8,  4,  1, 19, 17,  8,  4,  0,
       21, 20,  4,  4, 20, 21,  0, 16,  4,  0, 22, 20, 16,  4, 18,  0,  6,
        2,  4,  6,  0, 18, 22,  4,  6, 19, 18,  2,  4, 18, 19,  6, 22,  4,
        6, 23, 20, 22,  4, 20, 23,  6,  4,  4, 20,  0,  6, 22,  4,  6,  0,
       20,  4,  4, 13, 11,  5,  9,  4,  3, 11,  5,  7,  4,  3,  5, 11,  9,
        4, 12, 15,  5, 13,  4, 12,  5, 15,  7,  4, 11,  5, 15, 13,  4, 11,
       15,  5,  7,  4, 14,  7, 19, 23,  4,  6,  7, 19,  3,  4,  6, 19,  7,
       23,  4, 19, 10,  7, 14,  4, 19,  7, 10,  3,  4, 11,  7, 10, 14,  4,
       11, 10,  7,  3,  4,  1, 17,  2,  0,  4,  1,  2, 17, 19,  4, 18,  2,
       17,  0,  4, 18, 17,  2, 19,  4, 12, 13,  8, 24,  4, 13,  9,  8, 24,
        4,  9, 10,  8, 24,  4, 11, 10,  9, 24,  4, 11, 10, 24, 25,  4, 10,
       11, 14, 25,  4, 13, 11,  9, 24,  4, 10,  8, 24, 25,  4,  8, 12, 24,
       25,  4, 14, 12,  8, 25,  4, 10, 14,  8, 25,  4, 25, 14, 15, 11,  4,
       25, 15, 14, 12,  4, 13, 24, 15, 11,  4, 13, 15, 24, 12,  4, 24, 25,
       15, 11,  4, 24, 15, 25, 12,  4, 20, 21, 16, 26,  4, 21, 17, 16, 26,
        4, 17, 18, 16, 26,  4, 19, 18, 17, 26,  4, 19, 18, 26, 27,  4, 18,
       19, 22, 27,  4, 21, 19, 17, 26,  4, 18, 16, 26, 27,  4, 16, 20, 26,
       27,  4, 22, 20, 16, 27,  4, 18, 22, 16, 27,  4, 27, 22, 23, 19,  4,
       27, 23, 22, 20,  4, 21, 26, 23, 19,  4, 21, 23, 26, 20,  4, 26, 27,
       23, 19,  4, 26, 23, 27, 20])

    assert np.all(np.isclose(tet.points, correct_points))
    assert np.array_equal(tet.cells, correct_cells)


def test_pymeshlab_boolean():
    a = pv.Box(quads=False)
    b = pv.Box(quads=False).translate([0, 0, 0.5], inplace=False)

    unioned_mesh = pymeshlab_boolean([a,b], operation="Union")

    # p = pv.Plotter()
    # p.add_mesh(unioned_mesh, cmap=cm.get_cmap("Set1"), style="wireframe")
    # p.show()

    points = np.array([[-1., -1., -1.], [-1., -1., -0.5], [-1., -1., 1.], [-1., -1., 1.5],
                       [-1., -0.5, -0.5], [-1., 0.5, 1.], [-1., 1., -1.], [-1., 1., -0.5],
                       [-1., 1., 1.], [-1., 1., 1.5], [-0.5, -1., -0.5], [-0.5, 1., 1.],
                       [0.5, -1., 1.], [0.5, 1., -0.5], [1., -1., -1.], [1., -1., -0.5],
                       [1., -1., 1.], [1., -1., 1.5], [1., -0.5, 1.], [1., 0.5, -0.5],
                       [1., 1., -1.], [1., 1., -0.5], [1., 1., 1.], [1., 1., 1.5]])
    faces = np.array([ 3,  0,  1,  4,  3,  1,  0, 10,  3,  6,  0,  4,  3, 14,  0,  6,  3,
        0, 14, 10,  3,  1,  2,  5,  3,  2,  1, 12,  3,  4,  1,  5,  3, 12,
        1, 10,  3,  5,  2,  3,  3, 12,  3,  2,  3,  5,  3,  9,  3,  3, 23,
        9,  3, 17,  3, 12,  3,  3, 17, 23,  3,  4,  5,  8,  3,  7,  6,  4,
        3,  7,  4,  8,  3,  8,  5,  9,  3,  6,  7, 13,  3, 20,  6, 13,  3,
       14,  6, 20,  3,  7,  8, 13,  3,  8,  9, 11,  3,  8, 11, 13,  3, 23,
       11,  9,  3, 16, 12, 10,  3, 10, 14, 15,  3, 10, 15, 16,  3, 13, 11,
       21,  3, 11, 22, 21,  3, 11, 23, 22,  3, 17, 12, 16,  3, 13, 21, 20,
        3, 19, 15, 14,  3, 19, 14, 20,  3, 16, 15, 19,  3, 18, 17, 16,  3,
       18, 16, 19,  3, 18, 23, 17,  3, 21, 18, 19,  3, 21, 22, 18,  3, 22,
       23, 18,  3, 20, 21, 19])

    assert np.array_equal(unioned_mesh.points, points)
    assert np.array_equal(unioned_mesh.faces, faces)


def test_dif_any_intersecting():
    intersecting_a = pv.Box(quads=False).translate([-1, 0, 0], inplace=False)
    intersecting_b = pv.Box(quads=False)
    not_shared = pv.Box(quads=False).translate([4, 0, 0], inplace=False)

    mesh_arrays = [mesh for mesh in [intersecting_a, intersecting_b, not_shared]]

    diffed_meshes = dif_any_intersecting(mesh_arrays)

    # p = pv.Plotter()
    # cmap = cm.get_cmap("Set1")
    # for i, mesh in enumerate(diffed_meshes):
    #     p.add_mesh(mesh, style="wireframe", color=cmap(i))
    # p.show()

    np.testing.assert_almost_equal(diffed_meshes[0].volume, 4)
    np.testing.assert_almost_equal(diffed_meshes[1].volume, 8)
    np.testing.assert_almost_equal(diffed_meshes[2].volume, 8)


def test_dif_any_intersecting_2():
    a = pv.Box(quads=False)
    a_repeat = pv.Box(quads=False)
    b = pv.Box(quads=False).translate([4, 0, 0], inplace=False)

    meshes = [mesh for mesh in [a, a_repeat, b]]

    try:
        diffed = dif_any_intersecting(meshes)
    except ValueError as e:
        assert e.args[0] == 'Difference operation returned None. Ensure meshes do not overlap completely'

