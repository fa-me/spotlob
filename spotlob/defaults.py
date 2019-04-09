from IPython.display import display

import os.path

from .widget import SpotlobNotebookGui
from .process_opencv import SimpleReader, GreyscaleConverter,\
    GaussianPreprocess, OtsuThreshold, PostprocessNothing, \
    ContourFinderSimple, FeatureFormFilter
from .pipeline import Pipeline
from .preview import MatplotlibPreviewScreen
from .register import ProcessRegister
from .analyse_circle import CircleAnalysis
from .analyse_line import LineAnalysis
from .spim import Spim


def default_pipeline(mode="circle"):
    """Gives a pipeline which works for many cases and can be used as a
    starting point for further tuning or as default for the GUI notebook.
    By default, features with an area smaller than 500 pixels are ignored.

    Parameters
    ----------
    mode : str, optional
        this defines the way in which detected features will be evaluated,
        either "cirlce" or "line".
        * as `line` the linewidth is calculated
        * as `circle` an ellipse is fitted and by default features that touch
        the edge of the image get ignored
        (the default is "circle")

    Returns
    -------
    Pipeline
        the pipeline including default parameters
    """

    if mode == "circle":
        feature_form_filter = FeatureFormFilter(500, 0, True)
        analysis = CircleAnalysis()
    elif mode == "line":
        feature_form_filter = FeatureFormFilter(500, 0, False)
        analysis = LineAnalysis()

    return Pipeline([SimpleReader(),
                     GreyscaleConverter(),
                     GaussianPreprocess(1),
                     OtsuThreshold(),
                     PostprocessNothing(),
                     ContourFinderSimple(),
                     feature_form_filter,
                     analysis])


def make_gui(spim_or_filepath):
    """Creates a :any:`SpotlobNotebookGui` object which opens
    a given :any:`Spim` or image file for preview editing

    PARAMETERS
    ----------
    spim_or_filepath : Spim or str
        Spim (should be cached) or filepath of an image file
        to be loaded
    
    RETURNS
    -------
    SpotlobNotebookGui
        GUI object that can be displayed using the :any:`show_gui`
        function within a jupyter notebook
    """
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
    """Display a :any:`SpotlobNotebookGui` object.
    Run the `%matplotlib notebook` magic command to get live preview

    PARAMETERS
    ----------
    gui : SpotlobNotebookGui
        Widget and preview screen, as created by the :any:`make_gui`
        function
    """
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
