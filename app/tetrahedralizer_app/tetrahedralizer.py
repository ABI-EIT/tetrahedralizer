import app
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
import json
import pyvista as pv
from typing import List, Tuple
import itertools

import pyvista_tools
from pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d, remove_shared_faces
import numpy as np
from matplotlib import cm
import os
import pathlib

"""
App to automate the steps of creating a tetrahedralized mesh from one outer surface and several inner surfaces
"""


def main():
    # Load config
    config_filename = "conf.json"
    with open(config_filename, "r") as f:
        config = json.load(f)

    output_directory = config["output_directory"]
    output_suffix = config["output_suffix"]
    output_extension = config["output_extension"]
    mesh_repair_kwargs = config["mesh_repair_kwargs"]
    gmsh_options = config["gmsh_options"]

    # Select files
    Tk().withdraw()
    filename = askopenfilename(title="Select outer mesh")
    if filename == "":
        return
    Tk().destroy()

    Tk().withdraw()
    filenames = askopenfilenames(title="Select inner meshes")
    if len(filenames) == 0:
        return
    Tk().destroy()

    # Load files
    outer_mesh = pv.read(filename)
    inner_meshes = [pv.read(filename) for filename in filenames]

    # Run tetrahedralizer
    combined = preprocess_and_tetrahedralize(outer_mesh, inner_meshes, mesh_repair_kwargs, gmsh_options)

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    filename = f"{output_directory}/{output_filename}{output_extension}"
    combined.save(f"{filename}")

    # Plot result
    p = pv.Plotter()
    cmap = cm.get_cmap("Accent")
    p.add_mesh(combined, opacity=0.15, cmap=cmap, show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = combined.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", cmap=cmap, show_edges=True)

    p.add_plane_widget(plane_func, assign_to_axis="z")
    p.add_title("Combined Tetrahedralized Lung Sections")
    p.show()


def preprocess_and_tetrahedralize(outer_mesh: pv.DataSet, inner_meshes: List[pv.DataSet], mesh_repair_kwargs: dict,
                                  gmsh_options: dict) -> pv.UnstructuredGrid:
    print("Fixing...")
    # Fix all inputs
    fixed_meshes = []
    for mesh in [outer_mesh, *inner_meshes]:
        fixed_meshes.append(app.fix_mesh(mesh, mesh_repair_kwargs)[0])

    # Convert to arrays for boolean process
    fixed_mesh_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in fixed_meshes]

    print("Booleaning...")
    # Check all pairs of inner meshes for intersections and create:
    # # List of meshes where intersecting sets are replaced with a union
    # # List of meshes where intersecting pairs are replaced with a diffed and an original
    unioned_meshes = union_any_intersecting(fixed_mesh_arrays[1:])
    diffed_meshes = dif_any_intersecting(fixed_mesh_arrays[1:])

    # Convert back to pyvista
    pv_unioned_meshes = [pv.PolyData(mesh[0], pyvista_faces_to_1d(mesh[1])) for mesh in unioned_meshes]
    pv_diffed_meshes = [pv.PolyData(mesh[0], pyvista_faces_to_1d(mesh[1])) for mesh in diffed_meshes]

    # Fix booleaned meshes
    fixed_unioned = [app.fix_mesh(mesh, mesh_repair_kwargs)[0] for mesh in pv_unioned_meshes]
    fixed_diffed = [app.fix_mesh(mesh, mesh_repair_kwargs)[0] for mesh in pv_diffed_meshes]

    print("Combining...")
    # Remove shared faces to form inner hole
    combined_unioned = remove_shared_faces(fixed_unioned)
    fixed_combined = app.fix_mesh(combined_unioned)[0]
    fixed_combined_arrays = (fixed_combined.points, pyvista_faces_to_2d(fixed_combined.faces))

    print("Tetrahedralizing...")
    # Tetrahedralize outer mesh with hole, then convert to pyvista
    nodes, elements = app.gmsh_tetrahedralize([fixed_mesh_arrays[0], fixed_combined_arrays], gmsh_options)
    outer_tetrahedralized = pyvista_tools.pyvista_tetrahedral_mesh_from_arrays(nodes, elements[1])

    # Tetrahedralize each inner mesh, then convert to pyvista
    inner_tetrahedralized = []
    fixed_diffed_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in fixed_diffed]
    for mesh in fixed_diffed_arrays:
        nodes, elements = app.gmsh_tetrahedralize([mesh], gmsh_options)
        inner_tetrahedralized.append(
            pyvista_tools.pyvista_tetrahedral_mesh_from_arrays(nodes, elements[1]))

    # Combine result
    meshes = [outer_tetrahedralized, *inner_tetrahedralized]
    for i, mesh in enumerate(meshes):
        mesh.cell_data["Scalar"] = np.asarray([i % len(meshes)] * mesh.n_cells)
    blocks = pv.MultiBlock(meshes)
    combined = blocks.combine()

    return combined


def union_any_intersecting(meshes: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    intersection_sets = []

    # Iterate through all pairs and create sets of intersecting meshes
    for (index_a, mesh_a), (index_b, mesh_b) in itertools.combinations(enumerate(meshes), 2):
        # If they intersect
        if app.pymeshlab_boolean((mesh_a, mesh_b), operation="Intersection") is not None:
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
        union_result = app.pymeshlab_boolean((meshes[set_list[0]], meshes[set_list[1]]), operation="Union")
        for index in set_list[2:]:
            union_result = app.pymeshlab_boolean((union_result, meshes[index]), operation="Union")
        unioned_meshes.append(union_result)

    # Put back in any that weren't unioned
    unioned_indices = list(itertools.chain(*intersection_sets))
    for index, mesh in enumerate(meshes):
        if index not in unioned_indices:
            unioned_meshes.append(mesh)

    return unioned_meshes


def dif_any_intersecting(meshes: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    intersection_list = []
    dif_pairs = []

    # Iterate through all pairs and create sets of intersecting meshes
    index_b: object
    for (index_a, mesh_a), (index_b, mesh_b) in itertools.combinations(enumerate(meshes), 2):
        # If they intersect
        if app.pymeshlab_boolean((mesh_a, mesh_b), operation="Intersection") is not None:
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
        diffed_meshes.append(app.pymeshlab_boolean((meshes[pair[0]], meshes[pair[1]]), operation="Difference"))

    # Put back in any that weren't diffed
    diffed_indices = [pair[0] for pair in dif_pairs]
    for index, mesh in enumerate(meshes):
        if index not in diffed_indices:
            diffed_meshes.append(mesh)

    return diffed_meshes


if __name__ == "__main__":
    main()
