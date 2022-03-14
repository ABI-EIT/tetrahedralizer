import meshio
import pyvista as pv
import numpy as np
import tempfile
from typing import Tuple, Dict
import os

def main():
    output_directory = "output"
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

    mesh = pv.PolyData(mesh_a_verts, mesh_a_faces)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # mesh.save(r"output\test.ply")
    mesh = meshio_convert(mesh.save, filename_arg = "filename", extension=".ply")
    pass

def meshio_convert(save_func: callable, filename_arg: str, extension: str = None, args: Tuple = None, kwargs: Dict = None) -> meshio.Mesh:
    '''
    Beginings of a meshio convert idea. Sometimes you may want to:
        - convert an object to meshio automatically,
        - convert meshio to an object automatically
        - convert between one object and another
            - (or convert between object type a and object type b when type b only has a load from file function)

    Since meshio only works on file formats, it has no knowledge of object representations. To make an object converter
    would mean implementing each type from scratch.

    You can read and write files to change formats, but this may not be ideal as it could cause filename collisions,
    especially in multithreaded applications. Tempfile provides a solution. We could create wrappers like these to
    go through tempfiles to make conversions without the user having to know about it.

    This wrapper is the one that goes object to meshio since it takes a writer (save func). To go meshio to object,
    you would need reader. To convert objects you would need both and use meshio internally as an intermediate object.

    How can we make this less clunky to use? We could have a factory. And make these return functions... it's a bit tricky
    because you do need to know the filename argument. maybe we could guess it most of the time though

    Parameters
    ----------
    save_func
    filename_arg
    extension
    args
    kwargs

    Returns
    -------

    '''
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    temp_file_name = temp_file.name
    kwargs[filename_arg] = temp_file_name

    save_func(*args, **kwargs)
    temp_file.close()

    mesh = meshio.read(temp_file_name)
    os.remove(temp_file_name)

    return mesh



if __name__ == "__main__":
    main()
