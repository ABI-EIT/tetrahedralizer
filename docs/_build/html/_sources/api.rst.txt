=============
API Reference
=============

Tetrahedralizer API
-------------------

.. automodule:: tetrahedralizer.mesh_lib
    :members: fix_mesh, pymeshlab_boolean, gmsh_tetrahedralize, preprocess_and_tetrahedralize

.. automodule:: tetrahedralizer.pyvista_tools
    :members: remove_shared_faces


Pyvista Tools
-------------

.. automodule:: tetrahedralizer.pyvista_tools
    :members: pyvista_faces_to_2d, pyvista_faces_to_1d, extract_faces_with_edges, rewind_faces_to_normals,
              select_intersecting_triangles, refine_surface, dihedral_angle, repeatedly_fill_holes,
              fill_holes, triangulate_loop_with_stitch, triangulate_loop_with_nearest_neighbors
