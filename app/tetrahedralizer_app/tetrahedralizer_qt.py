import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, QProcess
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
Ui_MainWindow, QMainWindow = uic.loadUiType("layout/tetrahedralizer_layout.ui")
import pyvista as pv
import json
from app import preprocess_and_tetrahedralize
from pyvistaqt import QtInteractor
from matplotlib import cm
import os
import pathlib
from typing import Tuple
import adv_prodcon
from queue import Queue

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.redirect_std_out()

        self.outer_mesh = None
        self.inner_meshes = []
        self.tetrahedralized_mesh = None
        self.outer_mesh_filename = None
        self.inner_meshes_filenames = []
        self.worker = None

        config_filename = "conf.json"
        with open(config_filename, "r") as f:
            config = json.load(f)

        self.output_directory = config["output_directory"]
        self.output_suffix = config["output_suffix"]
        self.output_extension = config["output_extension"]
        self.mesh_repair_kwargs = config["mesh_repair_kwargs"]
        self.gmsh_options = config["gmsh_options"]

        self.outer_mesh_tool_button.clicked.connect(self.get_outer_mesh)
        self.inner_meshes_tool_button.clicked.connect(self.get_inner_meshes)
        self.tetrahedralize_button.clicked.connect(self.run_tetrahedralization)

        vlayout = QtWidgets.QVBoxLayout()
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        self.frame.setLayout(vlayout)
        vlayout.setContentsMargins(0, 0, 0, 0)

        self.plotter.add_axes()
        self.plotter.show()

    def get_outer_mesh(self):
        try:
            Tk().withdraw()
            filename = askopenfilename(title="Select outer mesh")
            Tk().destroy()
            self.outer_mesh = pv.read(filename)
        except (FileNotFoundError, ValueError) as e:
            print(e)
            return

        self.outer_mesh_filename = filename
        self.outer_mesh_line_edit.setText(filename)
        self.plot_surfaces()

    def get_inner_meshes(self):
        try:
            Tk().withdraw()
            filenames = askopenfilenames(title="Select inner meshes")
            Tk().destroy()
            self.inner_meshes = [pv.read(filename) for filename in filenames]
        except (FileNotFoundError,) as e:
            print(e)
            return

        self.inner_meshes_filenames = filenames
        self.inner_meshes_line_edit.setText(" ".join(filenames))
        self.plot_surfaces()

    def plot_surfaces(self):
        self.plotter.clear()
        meshes = []
        if self.outer_mesh is not None:
            meshes.append(self.outer_mesh)
        if self.inner_meshes:
            meshes.extend(self.inner_meshes)

        cmap = cm.get_cmap("Set1")  # Choose a qualitative colormap to distinguish meshes
        for i, mesh in enumerate(meshes):
            self.plotter.add_mesh(mesh, style="wireframe", opacity=0.5, color=cmap(i)[:-1], line_width=2)
        self.plotter.add_axes()

    def plot_volumes(self):
        if self.tetrahedralized_mesh is not None:
            self.plotter.clear()
            cmap = cm.get_cmap("Accent")
            self.plotter.add_mesh(self.tetrahedralized_mesh, opacity=0.15, cmap=cmap, show_edges=True, edge_color="gray")

            def plane_func(normal, origin):
                slc = self.tetrahedralized_mesh.slice(normal=normal, origin=origin)
                self.plotter.add_mesh(slc, name="slice", cmap=cmap, show_edges=True)

            self.plotter.add_plane_widget(plane_func, assign_to_axis="z")
            self.plotter.add_axes()

    def run_tetrahedralization(self):
        if self.outer_mesh is None or not self.inner_meshes:
            self.textEdit.append("Please select input meshes")
            return

        self.textEdit.append("Running tetrahedralization...")

        self.worker = Worker(self.outer_mesh, self.inner_meshes, self.mesh_repair_kwargs, self.gmsh_options)
        dummy_subscriber = adv_prodcon.Consumer()
        self.worker.set_subscribers([dummy_subscriber.get_work_queue()])
        self.worker.finished.connect(self.after_tetrahedralization)
        self.worker.std_out.connect(self.append_text)
        self.worker.start_new()

    def after_tetrahedralization(self, result: pv.UnstructuredGrid, error: str):

        if error != "":
            self.textEdit.append(error)
            return

        self.tetrahedralized_mesh = result
        self.textEdit.append("Tetrahedralization complete")

        try:
            if not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)

            output_filename = ""
            for filename in [self.outer_mesh_filename, *self.inner_meshes_filenames]:
                mesh_path = pathlib.Path(filename)
                output_filename += f"{mesh_path.stem}_"
            output_filename += self.output_suffix

            filename = f"{self.output_directory}/{output_filename}{self.output_extension}"
            self.tetrahedralized_mesh.save(f"{filename}")
            self.textEdit.append(f"Saved output mesh in {filename}")
        except Exception as e:
            self.textEdit.append(str(e))

        self.plot_volumes()

    @pyqtSlot(str)
    def append_text(self, text):
        self.textEdit.append(text)

    def redirect_std_out(self):
        self.queue = Queue()
        sys.stdout = WriteStream(self.queue)
        self.thread = QThread()
        self.receiver = Receiver(self.queue)
        self.receiver.signal.connect(self.append_text)
        self.receiver.moveToThread(self.thread)
        self.thread.started.connect(self.receiver.run)
        self.thread.start()


class Worker(adv_prodcon.Producer, QObject):
    finished = pyqtSignal((pv.UnstructuredGrid, str))
    std_out = pyqtSignal(str)

    def __init__(self, outer_mesh, inner_meshes, mesh_repair_kwargs, gmsh_options):
        super(adv_prodcon.Producer, self).__init__()
        super(QObject, self).__init__()
        self.work_kwargs = {"outer_mesh": outer_mesh, "inner_meshes": inner_meshes,
                            "mesh_repair_kwargs": mesh_repair_kwargs, "gmsh_options": gmsh_options}

    @staticmethod
    def on_start(state, message_pipe, *args, **kwargs):
        class writer(object):
            def __init__(self, pipe):
                self.pipe = pipe

            def write(self, text):
                self.pipe.send(text)

            def flush(self):
                pass

        sys.stdout = writer(message_pipe)

    @staticmethod
    def work(on_start_result, state, message_pipe, *args, **kwargs):
        message_pipe.send("Started")
        try:
            tetrahedralized_mesh = preprocess_and_tetrahedralize(kwargs["outer_mesh"], kwargs["inner_meshes"],
                                                                 kwargs["mesh_repair_kwargs"], kwargs["gmsh_options"])
            return tetrahedralized_mesh, ""
        except Exception as e:
            return pv.UnstructuredGrid(), str(e)

    def on_message_ready(self, message):
        if message == "Started":
            self.set_stopped()
        else:
            self.std_out.emit(message)


    def on_result_ready(self, result):
        self.finished.emit(result[0], result[1])



# The new Stream Object which replaces the default stream associated with sys.stdout
# This object just puts data in a queue!
class WriteStream(object):
    def __init__(self,queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

# A QObject (to be run in a QThread) which sits waiting for data to come through a Queue.Queue().
# It blocks until data is available, and one it has got something from the queue, it sends
# it to the "MainThread" by emitting a Qt Signal
class Receiver(QObject):
    signal = pyqtSignal(str)

    def __init__(self,queue,*args,**kwargs):
        QObject.__init__(self,*args,**kwargs)
        self.queue = queue

    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.signal.emit(text)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setWindowTitle("Tetrahedralizer")
    dw = QtWidgets.QDesktopWidget()
    main_window.show()

    app.exec()
