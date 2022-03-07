from combine_lobes import select_faces_using_points, select_shared_points, select_shared_faces, pyvista_faces_to_2d, \
    pyvista_faces_to_1d, select_points_in_faces, merge_surfaces_removing_shared_faces

import numpy as np
import pyvista as pv

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

mesh_a_faces = np.hstack([[3, 0, 1, 5], [3, 0, 5, 4],
                          [3, 1, 3, 7], [3, 1, 7, 5],
                          [3, 3, 2, 6], [3, 3, 6, 7],
                          [3, 2, 0, 4], [3, 2, 4, 6],
                          [3, 4, 5, 9], [3, 5, 7, 9], [3, 7, 6, 9], [3, 6, 4, 9],
                          [3, 0, 1, 8], [3, 1, 3, 8], [3, 3, 2, 8], [3, 2, 0, 8]])

mesh_a = pv.PolyData(mesh_a_verts, mesh_a_faces)

mesh_b_verts = np.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [-1, 0, 0],
                         [-1, 0, 1],
                         [-1, 1, 0],
                         [-1, 1, 1],
                         [0, 0.5, 0.5]], dtype=float)

mesh_b_faces = np.hstack([[3, 0, 1, 5], [3, 0, 5, 4],
                          [3, 1, 3, 7], [3, 1, 7, 5],
                          [3, 3, 2, 6], [3, 3, 6, 7],
                          [3, 2, 0, 4], [3, 2, 4, 6],
                          [3, 4, 6, 7], [3, 4, 7, 5],
                          [3, 0, 1, 8], [3, 1, 3, 8], [3, 3, 2, 8], [3, 2, 0, 8]])

mesh_b = pv.PolyData(mesh_b_verts, mesh_b_faces)

mesh_c_verts = np.array([[1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0],
                         [1, 1, 1],
                         [2, 0, 0],
                         [2, 0, 1],
                         [2, 1, 0],
                         [2, 1, 1],
                         [1, 0.5, 0.5]], dtype=float)

mesh_c_faces = np.hstack([[3, 0, 1, 5], [3, 0, 5, 4],
                          [3, 1, 3, 7], [3, 1, 7, 5],
                          [3, 3, 2, 6], [3, 3, 6, 7],
                          [3, 2, 0, 4], [3, 2, 4, 6],
                          [3, 4, 6, 7], [3, 4, 7, 5],
                          [3, 0, 1, 8], [3, 1, 3, 8], [3, 3, 2, 8], [3, 2, 0, 8]])

mesh_c = pv.PolyData(mesh_c_verts, mesh_c_faces)

shared_faces_ab = ([12, 13, 14, 15], [10, 11, 12, 13])
shared_points_ab = ([0, 1, 2, 3, 8], [0, 1, 2, 3, 8])


def test_select_faces_using_points():
    faces = select_faces_using_points(mesh_a, shared_points_ab[0])
    assert faces == shared_faces_ab[0]


def test_select_shared_points():
    shared_points = select_shared_points(mesh_a, mesh_b)
    assert shared_points == shared_points_ab


def test_select_shared_faces():
    shared_faces = select_shared_faces(mesh_a, mesh_b)
    assert shared_faces == shared_faces_ab


def test_pyvista_faces_to_2d():
    mesh_b_faces_2d = np.array([[0, 1, 5], [0, 5, 4], [1, 3, 7], [1, 7, 5], [3, 2, 6], [3, 6, 7], [2, 0, 4], [2, 4, 6],
                                [4, 6, 7], [4, 7, 5], [0, 1, 8], [1, 3, 8], [3, 2, 8], [2, 0, 8]])

    faces_2d = pyvista_faces_to_2d(mesh_b.faces)

    assert np.array_equal(mesh_b_faces_2d, faces_2d)


def test_pyvista_faces_to_1d():
    mesh_b_faces_2d = np.array([[0, 1, 5], [0, 5, 4], [1, 3, 7], [1, 7, 5], [3, 2, 6], [3, 6, 7], [2, 0, 4], [2, 4, 6],
                                [4, 6, 7], [4, 7, 5], [0, 1, 8], [1, 3, 8], [3, 2, 8], [2, 0, 8]])

    faces_1d = pyvista_faces_to_1d(mesh_b_faces_2d)
    assert np.array_equal(faces_1d, mesh_b.faces)


def test_select_points_in_faces():
    exclusive_points = [8]
    test_points = [0, 1, 2, 3, 8]
    test_faces = [12, 13, 14, 15]

    points_in_test_faces = select_points_in_faces(mesh_a, test_points, test_faces, exclusive=True)

    assert np.array_equal(points_in_test_faces, exclusive_points)


def test_merge_surfaces_removing_shared_faces():
    a_merged_points = mesh_a_verts[:-2]
    a_merged_faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh_a_faces)[:-8])
    a_merged = pv.PolyData(a_merged_points, a_merged_faces)
    b_merged_points = mesh_b_verts[:-1]
    b_merged_faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh_b_faces)[:-4])
    b_merged = pv.PolyData(b_merged_points, b_merged_faces)
    c_merged_points = mesh_c_verts[:-1]
    c_merged_faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh_c_faces)[:-4])
    c_merged = pv.PolyData(c_merged_points, c_merged_faces)
    a_b_c_merged = pv.MultiBlock([a_merged, b_merged, c_merged]).combine(merge_points=True, tolerance=1e-05)

    merged = merge_surfaces_removing_shared_faces([mesh_a, mesh_b, mesh_c])

    assert np.array_equal(a_b_c_merged.points, merged.points)
    assert np.array_equal(a_b_c_merged.cells, merged.cells)


def main():
    p = pv.Plotter()
    p.add_mesh(mesh_a, style="wireframe")
    p.add_mesh(mesh_b, style="wireframe")
    p.add_mesh(mesh_c, style="wireframe")
    p.show()


if __name__ == "__main__":
    main()
