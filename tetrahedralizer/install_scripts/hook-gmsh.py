import gmsh
import platform
import os
from ctypes.util import find_library

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

libpath = os.path.join(libdir, libname)
if not os.path.exists(libpath):
    libpath = os.path.join(libdir, "Lib", libname)
if not os.path.exists(libpath):
    libpath = os.path.join(moduledir, libname)
if not os.path.exists(libpath):
    if platform.system() == "Windows":
        libpath = find_library(f"gmsh-{version}")
        if not libpath:
            libpath = find_library("gmsh")
    else:
        libpath = find_library("gmsh")

binaries = [(str(libpath), '.')]
