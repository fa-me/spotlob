from IPython.display import display

import widget
import process_opencv
import pipeline
import preview
import register


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
    pipe = default_pipeline(start_filepath)
    preview_screen = preview.MatplotlibPreviewScreen()
    gui = widget.SpotlobNotebookGui(pipe, preview_screen)
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
