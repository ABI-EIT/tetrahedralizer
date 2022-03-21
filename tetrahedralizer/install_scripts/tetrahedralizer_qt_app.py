import PyInstaller.__main__
import tetrahedralizer
from pathlib import Path
import shutil
import os


def install():
    package_dir = str(Path(tetrahedralizer.__file__).parent)
    workpath = r".\build"
    distpath = r"."
    specpath = r"."

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

    shutil.rmtree(workpath)
    os.remove(specpath + r"\tetrahedralizer_qt.spec")


if __name__ == "__main__":
    install()
