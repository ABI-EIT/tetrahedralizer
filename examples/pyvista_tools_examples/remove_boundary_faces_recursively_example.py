import pyvista as pv
from tetrahedralizer.pyvista_tools import remove_boundary_faces_recursively, identify_neighbors, remove_shared_faces


def main():
    # Create example mesh
    half_sphere = pv.Sphere().clip()
    full_sphere = pv.Sphere(center=(-0.5, 0, 0))
    union = half_sphere.boolean_union(full_sphere)
    intersection = half_sphere.boolean_intersection(full_sphere)
    example_mesh = union.merge(intersection)
    example_mesh = pv.PolyData(example_mesh.points, example_mesh.faces)

    example_mesh.plot(show_edges=True)

    # Remove boundary edges
    boundary_removed = remove_boundary_faces_recursively(example_mesh)

    # Plot result
    shared_faces_removed = remove_shared_faces([example_mesh, boundary_removed])
    p = pv.Plotter()
    p.add_mesh(boundary_removed, show_edges=True, label="Boundary Faces Removed")
    p.add_mesh(shared_faces_removed[0], color="red", show_edges=True, opacity=.2, label="Removed Faces")
    p.add_legend()
    p.show()


if __name__ == "__main__":
    main()
