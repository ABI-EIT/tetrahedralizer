import numpy as np
import os
import pathlib
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import gmsh
import json
import pyvista as pv
from typing import List, Tuple
import vtkmodules
from pyvista_tools import pyvista_faces_to_2d, pyvista_tetrahedral_mesh_from_arrays, pyvista_faces_by_dimension
import meshio


def main():
    config_filename = "conf.json"
    with open(config_filename, "r") as f:
        config = json.load(f)

    output_directory = config["output_directory"]
    output_suffix = config["output_suffix"]
    gmsh_options = config["gmsh_options"]

    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to gmsh")
    if len(filenames) == 0:
        return
    Tk().destroy()

    meshes = [pv.read(filename) for filename in filenames]
    mesh_arrays = [(mesh.points, pyvista_faces_to_2d(mesh.faces)) for mesh in meshes]

    nodes, elements = gmsh_tetrahedralize(mesh_arrays, gmsh_options)
    gmsh.finalize()

    mesh = pyvista_tetrahedral_mesh_from_arrays(nodes, elements[0], elements[1])

    # Save result
    output_filename = ""
    for filename in filenames:
        mesh_path = pathlib.Path(filename)
        output_filename += f"{mesh_path.stem}_"
    output_filename += output_suffix

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    mesh_triangles = mesh.cells_dict[vtkmodules.util.vtkConstants.VTK_TRIANGLE]
    mesh_tets = mesh.cells_dict[vtkmodules.util.vtkConstants.VTK_TETRA]
    meshio_faces = {"triangle": mesh_triangles, "quad": mesh_tets}
    m = meshio.Mesh(mesh.points, meshio_faces)
    m.write(f"{output_directory}/{output_filename}.ply")

    # # Plot result
    p = pv.Plotter()
    p.add_mesh(mesh, opacity=0.15, show_edges=True, edge_color="gray")

    def plane_func(normal, origin):
        slc = mesh.slice(normal=normal, origin=origin)
        p.add_mesh(slc, name="slice", show_edges=True)

    p.add_plane_widget(plane_func, assign_to_axis="z")
    p.add_title("Tetrahedralized Mesh")
    p.show()


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

    return nodes, elements


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
    if max_node == 0:
        max_node = 1
    node_tags = max_node + np.array(range(mesh[0].shape[0]))
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


if __name__ == "__main__":
    main()
