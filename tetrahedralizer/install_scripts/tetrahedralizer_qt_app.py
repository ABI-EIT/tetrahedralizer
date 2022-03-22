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
    package_dir = os.path.dirname(tetrahedralizer.__file__)
    working_dir = os.getcwd()
    workpath = os.path.join(working_dir, "build")
    distpath = os.path.join(working_dir, "tetrahedralizer_qt")
    specpath = working_dir

    app_path = os.path.join(package_dir, "app", "tetrahedralizer_qt_app", "tetrahedralizer_qt.py")
    layout_path = os.path.join(package_dir, "app", "tetrahedralizer_qt_app", "layout", "tetrahedralizer_layout.ui")
    conf_path = os.path.join(package_dir, "app", "tetrahedralizer_qt_app", "conf.json")
    hooks_dir = os.path.join(package_dir, "install_scripts")

    # Remove the distribution directory manually, since pyinstaller doesn't know about everything we put there
    if os.path.exists(distpath):
        shutil.rmtree(distpath)

    sep = os.pathsep
    PyInstaller.__main__.run([
        app_path,
        "--add-data=" + layout_path + sep + "layout",
        "--add-data=" + conf_path + sep + ".",
        "--hidden-import=vtkmodules",
        "--hidden-import=vtkmodules.vtkFiltersGeneral",
        "--collect-all=pymeshlab",
        "--additional-hooks-dir=" + hooks_dir,
        "--windowed",
        "--workpath=" + workpath,
        "--distpath=" + distpath,
        "--onedir"
    ])

    # Move conf.json up one level to get it out of the mess (as matched in tetrahedralizer_qt.py)
    shutil.move(os.path.join(distpath, "tetrahedralizer_qt", "conf.json"), distpath)

    # Get the example meshes
    shutil.copytree(os.path.join(package_dir, "app", "example_meshes"), os.path.join(distpath, "example_meshes"))
    os.remove(os.path.join(distpath, "example_meshes", "__init__.py"))

    # Create a shortcut to the pyinstaller exe and place it one level up
    short = make_shortcut(" ", name="tetrahedralizer_qt", terminal=False, startmenu=False,
                      executable=os.path.join(distpath, "tetrahedralizer_qt","tetrahedralizer_qt.exe"))
    shutil.move(os.path.join(short.desktop_dir, short.target), distpath)

    # Get rid of pyinstaller working files
    shutil.rmtree(workpath)
    os.remove(os.path.join(specpath,"tetrahedralizer_qt.spec"))


if __name__ == "__main__":
    install()
