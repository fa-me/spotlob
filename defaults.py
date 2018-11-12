from IPython.display import display

import widget
import process_opencv
import pipeline
import preview


def default_pipeline(start_filepath):
    return pipeline.Pipeline([process_opencv.SimpleReader(start_filepath),
                              process_opencv.GreyscaleConverter(),
                              process_opencv.GaussianPreprocess(3),
                              process_opencv.BinaryThreshold(100),
                              process_opencv.PostprocessNothing(),
                              process_opencv.ContourFinderSimple(),
                              process_opencv.FeatureFormFilter(4000, 0.98),
                              process_opencv.CircleAnalysis()])


def make_gui(start_filepath):
    gui = widget.SpotlobNotebookGui(
        default_pipeline(start_filepath), preview.MatplotlibPreviewScreen())
    return gui


def show_gui(gui):
    widgets = gui.make_widgets()
    gui.show_preview_screen(figsize=(8, 6))
    display(widgets)
    display(gui.run_button())
