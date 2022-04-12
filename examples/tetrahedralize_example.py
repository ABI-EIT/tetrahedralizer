import pyvista as pv
from tetrahedralizer.mesh_lib import preprocess_and_tetrahedralize
from matplotlib import cm
import os
import pathlib

"""
Example script for the tetrahedralizer library from the ABI-EIT github organization. This example runs the 
preprocess_and_tetrahedralize function on a set of example lung meshes.

Install the tetrahedralizer library using pip install git+https://github.com/ABI-EIT/tetrahedralizer

For full docs on tetrahedralizer, see: https://abi-eit.github.io/tetrahedralizer/_build/html/index.html
"""

config = {
  "output_suffix": "tetrahedralized",
  "output_directory": "output",
  "output_extension": ".vtu",
  "mesh_repair_kwargs": {},
  "gmsh_options": {
    "Mesh.MeshSizeMax": 10
  }
}

outer_mesh_filename = "./meshes/mock_lung/mock_torso.STL"

inner_mesh_filenames = ["./meshes/mock_lung/lower_lobe_of_left_lung_surface.stl",
                        "./meshes/mock_lung/upper_lobe_of_left_lung_surface.stl",
                        "./meshes/mock_lung/lower_lobe_of_right_lung_surface.stl",
                        "./meshes/mock_lung/middle_lobe_of_right_lung_surface.stl",
                        "./meshes/mock_lung/upper_lobe_of_right_lung_surface.stl"]


def main():
    output_directory = config["output_directory"]
    output_suffix = config["output_suffix"]
    output_extension = config["output_extension"]
    mesh_repair_kwargs = config["mesh_repair_kwargs"]
    gmsh_options = config["gmsh_options"]

    # Load files
    outer_mesh = pv.PolyData(pv.read(outer_mesh_filename))
    inner_meshes = [pv.PolyData(pv.read(filename)) for filename in inner_mesh_filenames]

    # Plot input meshes
    cmap = cm.get_cmap("Set1")  # Choose a qualitative colormap to distinguish meshes
    p = pv.Plotter()
    for i, mesh in enumerate([outer_mesh, *inner_meshes]):
        p.add_mesh(mesh, style="wireframe", opacity=0.5, color=cmap(i)[:-1], line_width=2)
    p.add_title("Input surfaces")
    p.show()

    # Run tetrahedralizer
    combined = preprocess_and_tetrahedralize(outer_mesh, inner_meshes, mesh_repair_kwargs, gmsh_options)

    # Save result
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_filename = ""
    for filename in inner_mesh_filenames:
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


if __name__ == "__main__":
    main()
