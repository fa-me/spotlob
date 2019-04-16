from IPython.display import display

import os.path

from .widget import SpotlobNotebookGui
from .process_opencv import SimpleReader, GreyscaleConverter,\
    GaussianPreprocess, OtsuThreshold, BinaryThreshold, PostprocessNothing, \
    ContourFinderSimple, FeatureFormFilter
from .pipeline import Pipeline
from .preview import MatplotlibPreviewScreen
from .register import PROCESS_REGISTER
from .analyze_circle import CircleAnalysis
from .analyze_line import LineAnalysis
from .spim import Spim


def default_pipeline(mode="circle", thresholding="auto"):
    """Gives a pipeline which works for many cases and can be used as a
    starting point for further tuning or as default for the GUI notebook.
    By default, features with an area smaller than 500 pixels are ignored.

    Parameters
    ----------
    mode : str, optional
        this defines the way in which detected features will be evaluated

        - as `line` the linewidth is calculated
        - as `circle` an ellipse is fitted and by default features that touch
          the edge of the image get ignored (the default is "circle")

    thresholding : str, optional
        - `auto` uses Otsu's thresholding algorithm
        - `simple` uses a fixed threshold value, 100 by default
          (the default is "auto")

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

    if thresholding == "auto":
        binarization = OtsuThreshold()
    elif thresholding == "simple":
        binarization = BinaryThreshold(100)

    return Pipeline([SimpleReader(),
                     GreyscaleConverter(),
                     GaussianPreprocess(1),
                     binarization,
                     PostprocessNothing(),
                     ContourFinderSimple(),
                     feature_form_filter,
                     analysis])


def make_gui(spim_or_filepath, mode="circle"):
    """Creates a :class:`~spotlob.SpotlobNotebookGui` object which opens
    a given :class:`~spotlob.Spim` or image file for preview editing

    PARAMETERS
    ----------
    spim_or_filepath : Spim or str
        Spim (should be cached) or filepath of an image file
        to be loaded

    mode : str
        "circle" or "line" depending on how you want the image to
        be evaluated

    RETURNS
    -------
    SpotlobNotebookGui
        GUI object that can be displayed using the
        :func:`~spotlob.defaults.show_gui` function within
        a jupyter notebook
    """
    if not isinstance(spim_or_filepath, Spim):
        if os.path.exists(spim_or_filepath):
            spim = Spim.from_file(spim_or_filepath, cached=True)
        else:
            raise FileNotFoundError("File %s not found" % spim_or_filepath)
    else:
        spim = spim_or_filepath

    pipe = default_pipeline()
    preview_screen = MatplotlibPreviewScreen()
    gui = SpotlobNotebookGui(pipe, preview_screen, spim)
    return gui


def show_gui(gui):
    """Display a :class:`~spotlob.SpotlobNotebookGui` object.
    Run the `%matplotlib notebook` magic command to get live preview

    PARAMETERS
    ----------
    gui : SpotlobNotebookGui
        Widget and preview screen, as created by the
        :func:`~spotlob.defaults.show_gui` function
    """
    widgets = gui.make_widgets()
    gui.show_preview_screen(figsize=(8, 6))
    display(widgets)
    display(gui.run_button())


def use_in(gui, register=None):
    """Use this as a decorator replace a process in a
    :class:`~spotlob.SpotlobNotebookGui` object

    PARAMETERS
    ----------
    gui : SpotlobNotebookGui
        GUI object to replace the process in

    RETURNS
    -------
    callable
        wrapper function
    """
    if register is None:
        register = PROCESS_REGISTER

    def wrapper(fn):
        process = register.available_processes[fn.__name__]

        # overwrite process at the given stage in pipeline of gui
        gui.pipeline = gui.pipeline.replaced_with(process)
        return fn
    return wrapper


def load_image(filepath, cached=False):
    """Create a :class:`~spotlob.Spim` from a filepath directly,
    using a default reader

    PARAMETERS
    ----------
    filepath : str
        Path to the image file
    cached : bool
        Wether or not references to previous Spim should be kept
        when processing this spim to newer versions

    RETURNS
    -------
    Spim
        Spim containing the image, at stage SpimStage.loaded
    """
    spim = Spim.from_file(filepath, cached=cached)
    reader = SimpleReader()
    return spim.read(reader)
