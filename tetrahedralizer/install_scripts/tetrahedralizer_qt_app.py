import PyInstaller.__main__
import tetrahedralizer
from pathlib import Path
import shutil
import os
from pyshortcuts import make_shortcut


def install():
    """
    Installation script for tetrahedralizer_qt.py. Bundles the app with pyinstaller, then tidies things up.
    """
    package_dir = str(Path(tetrahedralizer.__file__).parent)
    working_dir = os.getcwd()
    workpath = working_dir + r"\build"
    distpath = working_dir + r"\tetrahedralizer_qt"
    specpath = working_dir

    # Remove the distribution directory manually, since pyinstaller doesn't know about everything we put there
    if os.path.exists(distpath):
        shutil.rmtree(distpath)

    sep = os.pathsep
    PyInstaller.__main__.run([
        package_dir + r"\app\tetrahedralizer_qt_app\tetrahedralizer_qt.py",
        "--add-data=" + package_dir + r"\app\tetrahedralizer_qt_app\layout\tetrahedralizer_layout.ui" + sep + "layout",
        "--add-data=" + package_dir + r"\app\tetrahedralizer_qt_app\conf.json" + sep + ".",
        "--hidden-import=vtkmodules",
        "--hidden-import=vtkmodules.vtkFiltersGeneral",
        "--collect-all=pymeshlab",
        "--additional-hooks-dir=" + package_dir + r"\install_scripts",
        "--windowed",
        "--workpath=" + workpath,
        "--distpath=" + distpath,
        "--onedir"
    ])

    # Move conf.json up one level to get it out of the mess (as matched in tetrahedralizer_qt.py)
    shutil.move(distpath + r"\tetrahedralizer_qt\conf.json", distpath)

    # Get the example meshes
    shutil.copytree(package_dir + r"\app\example_meshes", distpath + r"\example_meshes")
    os.remove(distpath + r"\example_meshes\__init__.py")

    # Create a shortcut to the pyinstaller exe and place it one level up
    short = make_shortcut(" ", name="tetrahedralizer_qt", terminal=False, startmenu=False,
                      executable=distpath + r"\tetrahedralizer_qt\tetrahedralizer_qt.exe")
    shutil.move(Path(short.desktop_dir + "\\" + short.target), distpath)

    # Get rid of pyinstaller working files
    shutil.rmtree(workpath)
    os.remove(specpath + r"\tetrahedralizer_qt.spec")


if __name__ == "__main__":
    install()
