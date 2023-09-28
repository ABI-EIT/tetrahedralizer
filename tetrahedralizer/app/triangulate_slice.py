from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
import json
import pyvista as pv
from tetrahedralizer.mesh_lib import preprocess_and_tetrahedralize
from matplotlib import colormaps
import os
import pathlib
import triangle as tr
import numpy as np
from pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d
import pymeshlab
from tetrahedralizer.mesh_lib import fix_mesh
import itertools

"""
App to triangulate a slice of one or more 3D surface meshes
"""


def main():
    config = {
        "output_suffix": "triangulated",
        "output_directory": "output",
        "output_extension": ".stl",
        "slice_normal_axis": "x",
        "slice_position": 0.75,
        "mesh_repair_kwargs": {},
        "gmsh_options": {
            "Mesh.MeshSizeMax": 10
        }
    }

    axis_map = {"x": 0, "y": 1, "z": 2}
    planeaxis = axis_map[config["slice_normal_axis"]]
    slice_position = config["slice_position"]
    output_directory = config["output_directory"]
    output_suffix = config["output_suffix"]
    output_extension = config["output_extension"]
    mesh_repair_kwargs = config["mesh_repair_kwargs"]
    gmsh_options = config["gmsh_options"]

    # Select files
    Tk().withdraw()
    if not (filename := askopenfilename(title="Select outer mesh")):
        exit()
    Tk().destroy()

    Tk().withdraw()
    filenames = askopenfilenames(title="Select inner meshes")
    Tk().destroy()

    # Load files
    outer_mesh = pv.PolyData(pv.read(filename))
    inner_meshes = [pv.PolyData(pv.read(filename)) for filename in filenames]

    if not outer_mesh.is_manifold:
        outer_mesh = fix_mesh(outer_mesh)

    normal_axis_bound = np.array(outer_mesh.bounds).reshape((-1, 2))[planeaxis]

    bound_length = normal_axis_bound[1] - normal_axis_bound[0]
    offset_distance = slice_position * bound_length
    plane_offset = normal_axis_bound[0] + offset_distance

    sliced = pymeshlab_slice(outer_mesh, planeaxis=planeaxis, relativeto=2, planeoffset=plane_offset)

    triangulated = triangulate_slice_with_triangle(sliced)

    p = pv.Plotter()
    p.add_mesh(triangulated.extract_all_edges())
    p.add_mesh(outer_mesh, opacity=0.5)
    p.show_bounds(location="all")

    p.add_axes()
    p.show()

    pass

    # # Save result
    # if not os.path.exists(output_directory):
    #     os.mkdir(output_directory)
    #
    # output_filename = ""
    # for filename in filenames:
    #     mesh_path = pathlib.Path(filename)
    #     output_filename += f"{mesh_path.stem}_"
    # output_filename += output_suffix
    #
    # filename = f"{output_directory}/{output_filename}{output_extension}"
    # combined.save(f"{filename}")
    #
    # # Plot result
    # p = pv.Plotter()
    # cmap = colormaps["Accent"]
    # p.add_mesh(combined, opacity=0.15, cmap=cmap, show_edges=True, edge_color="gray")
    # p.add_title("Triangulated Slice")
    # p.show()


def pymeshlab_slice(mesh: pv.PolyData, **kwargs) -> pv.PolyData:
    mesh_arrays = (mesh.points, pyvista_faces_to_2d(mesh.faces), mesh.face_normals)
    ms = pymeshlab.MeshSet()
    ml_mesh = pymeshlab.Mesh(mesh_arrays[0], mesh_arrays[1], f_normals_matrix=mesh_arrays[2])
    ms.add_mesh(ml_mesh)

    ms.generate_polyline_from_planar_section(**kwargs)

    slice = (ms.mesh(1).vertex_matrix(), ms.mesh(1).edge_matrix())
    pv_slice = pv.PolyData(slice[0], lines=pyvista_faces_to_1d(slice[1]))

    return pv_slice


def triangulate_slice_with_triangle(slc: pv.PolyData) -> pv.PolyData:
    seg = slc.lines.reshape(-1, 3)[:, 1:]

    flat_axis = find_flat_axis(slc.points)
    if flat_axis == -1:
        raise ValueError("No flat axis found")

    points = np.delete(slc.points, flat_axis, axis=1)

    A = dict(vertices=np.array(points, dtype=np.float64), segments=seg)

    try:
        B = tr.triangulate(A, 'qp')
    except RuntimeError as e:
        B = tr.triangulate(A, "p")

    n_faces = B['triangles'].shape[0]
    triangles = np.hstack((np.ones((n_faces, 1), np.int32) * 3, B['triangles']))

    # back to 3D
    pts = np.empty((B['vertices'].shape[0], 3))

    c = itertools.count()
    for i in range(3):
        if i == flat_axis:
            pts[:, i] = slc.points[0, i]
        else:
            pts[:, i] = B["vertices"][:, next(c)]

    pd = pv.PolyData(pts, triangles, n_faces)
    return pd


def find_flat_axis(points: np.array, **allclose_kwargs) -> int:
    if not len(points.shape) == 2:
        raise ValueError("2D array expected")

    for i in range(points.shape[1]):
        if np.allclose(points[:, i], np.average(points[:, i]), **allclose_kwargs):
            return i

    return -1


if __name__ == "__main__":
    main()
