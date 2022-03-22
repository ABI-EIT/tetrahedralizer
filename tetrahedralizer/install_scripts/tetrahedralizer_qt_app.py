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

    # Todo: semicolons don't work on linux
    PyInstaller.__main__.run([
        package_dir + r"\app\tetrahedralizer_qt_app\tetrahedralizer_qt.py",
        "--add-data=" + package_dir + r"\app\tetrahedralizer_qt_app\layout\tetrahedralizer_layout.ui;layout",
        "--add-data=" + package_dir + r"\app\tetrahedralizer_qt_app\conf.json;.",
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

    # Create a shortcut to the pyinstaller exe and place it one level up
    s = make_shortcut(" ", name="tetrahedralizer_qt", terminal=False, startmenu=False,
                      executable=distpath + r"\tetrahedralizer_qt\tetrahedralizer_qt.exe")
    shutil.move(Path(s.desktop_dir + "\\" + s.target), distpath)

    # Get rid of pyinstaller working files
    shutil.rmtree(workpath)
    os.remove(specpath + r"\tetrahedralizer_qt.spec")


if __name__ == "__main__":
    install()
