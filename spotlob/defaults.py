from IPython.display import display

import os.path

import widget
import process_opencv
import pipeline
import preview
import register
import process_opencv
import analyse_opencv
from spim import *


def default_pipeline():
    return pipeline.Pipeline([process_opencv.SimpleReader(),
                              process_opencv.GreyscaleConverter(),
                              process_opencv.GaussianPreprocess(3),
                              process_opencv.BinaryThreshold(100),
                              process_opencv.PostprocessNothing(),
                              process_opencv.ContourFinderSimple(),
                              process_opencv.FeatureFormFilter(4000, 0.98),
                              analyse_opencv.CircleAnalysis()])


def make_gui(spim_or_filepath):
    try:
        assert os.path.exists(spim_or_filepath)
        spim = Spim.from_file(spim_or_filepath, cached=True)
    except AssertionError:
        spim = spim_or_filepath

    pipe = default_pipeline()
    preview_screen = preview.MatplotlibPreviewScreen()
    gui = widget.SpotlobNotebookGui(pipe, preview_screen, spim)
    return gui


def show_gui(gui):
    widgets = gui.make_widgets()
    gui.show_preview_screen(figsize=(8, 6))
    display(widgets)
    display(gui.run_button())


def use_in(gui):
    """decorator to tell gui to replace function"""
    def wrapper(fn):
        process = register.ProcessRegister.available_processes[fn.__name__]

        # overwrite process at the given stage in pipeline of gui
        gui.pipeline.process_stage_dict[process.input_stage] = process
        return fn
    return wrapper


def load_image(filepath, cached=False):
    spim = Spim.from_file(filepath, cached=cached)
    reader = process_opencv.SimpleReader()
    return spim.read(reader)
