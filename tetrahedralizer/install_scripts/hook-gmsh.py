from pathlib import Path
import gmsh
import platform
import os

moduledir = os.path.dirname(os.path.realpath(gmsh.__file__))
version = f"{gmsh.GMSH_API_VERSION_MAJOR}.{gmsh.GMSH_API_VERSION_MINOR}"

# Construct platform appropriate libpath. See gmsh.py
if platform.system() == "Windows":
    libname = f"gmsh-{version}.dll"
    libdir = os.path.dirname(moduledir)
elif platform.system() == "Darwin":
    libname = f"libgmsh.{version}.dylib"
    libdir = os.path.dirname(os.path.dirname(moduledir))
else:
    libname = f"libgmsh.so.{version}"
    libdir = os.path.dirname(os.path.dirname(moduledir))

libpath = Path(libdir) / libname
binaries = [(str(libpath), '.')]
