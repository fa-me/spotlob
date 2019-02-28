from IPython.display import display

import os.path

from .widget import SpotlobNotebookGui
from .process_opencv import SimpleReader, GreyscaleConverter,\
    GaussianPreprocess, BinaryThreshold, PostprocessNothing, \
    ContourFinderSimple, FeatureFormFilter
from .pipeline import Pipeline
from .preview import MatplotlibPreviewScreen
from .register import ProcessRegister
from .analyse_opencv import CircleAnalysis
from .spim import Spim


def default_pipeline():
    return Pipeline([SimpleReader(),
                     GreyscaleConverter(),
                     GaussianPreprocess(3),
                     BinaryThreshold(100),
                     PostprocessNothing(),
                     ContourFinderSimple(),
                     FeatureFormFilter(4000, 0.98),
                     CircleAnalysis()])


def make_gui(spim_or_filepath):
    try:
        assert os.path.exists(spim_or_filepath)
        spim = Spim.from_file(spim_or_filepath, cached=True)
    except AssertionError:
        spim = spim_or_filepath

    pipe = default_pipeline()
    preview_screen = MatplotlibPreviewScreen()
    gui = SpotlobNotebookGui(pipe, preview_screen, spim)
    return gui


def show_gui(gui):
    widgets = gui.make_widgets()
    gui.show_preview_screen(figsize=(8, 6))
    display(widgets)
    display(gui.run_button())


def use_in(gui):
    """decorator to tell gui to replace function"""
    def wrapper(fn):
        process = ProcessRegister.available_processes[fn.__name__]

        # overwrite process at the given stage in pipeline of gui
        gui.pipeline.process_stage_dict[process.input_stage] = process
        return fn
    return wrapper


def load_image(filepath, cached=False):
    spim = Spim.from_file(filepath, cached=cached)
    reader = SimpleReader()
    return spim.read(reader)
