import trimesh
import numpy as np
import os
import pathlib
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from matplotlib import cm


def main():
    # Select files
    Tk().withdraw()
    filenames = askopenfilenames(title="Select meshes to union")
    if len(filenames) == 0:
        return
    Tk().destroy()

    meshes = [trimesh.load(filename) for filename in filenames]

    # Needs a backend
    b = trimesh.boolean.boolean_automatic(meshes, "union")

    b.show()
    pass




if __name__ == "__main__":
    main()
