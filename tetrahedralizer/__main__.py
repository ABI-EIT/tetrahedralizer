import argparse
from tetrahedralizer.install_scripts import tetrahedralizer_qt_app

app_install_functions = {
    "tetrahedralizer_qt": tetrahedralizer_qt_app.install
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", help="Install a tetrahedralizer app", choices=["tetrahedralizer_qt"])
    args = parser.parse_args()

    app_install_functions[args.install]()


if __name__ == "__main__":
    main()
