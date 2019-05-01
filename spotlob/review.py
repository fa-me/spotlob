"""
This module provides a widget to browse through the
already applied detection, to see if everything worked
"""
import numpy as np

from ipywidgets import VBox, Button
from IPython.display import display, clear_output

from spotlob.preview import MatplotlibPreviewScreen
from spotlob.spim import Spim, SpimStage


def review_widgets(dataframe, pipeline):
    """Displays a review widget, that has a button
    to browse through all the files listed in a given
    dataframe and draw the result of the analysis contained
    in the given pipeline 

    Parameters
    ----------
    dataframe : pandas.Dataframe
        Result from a detection with pipeline
    pipeline : Pipeline
        The pipeline that the results have been created with
    """
    screen = MatplotlibPreviewScreen()

    load_process = pipeline.process_stage_dict[SpimStage.loaded-1]
    analysis_process = pipeline.process_stage_dict[SpimStage.analyzed-1]

    rows = dataframe.iterrows()

    def get_next(btn):
        _, row = next(rows)
        fp = row["filepath"]
        s0 = Spim.from_file(fp, cached=True).read(load_process)

        image = np.copy(s0.image)

        results_data = row.to_frame().T
        image_with_results = analysis_process.draw_results(image, results_data)

        if screen.fg_ax is None:
            screen.make_new(image_with_results)
        else:
            screen.update(image_with_results)

    rb = Button(description="next")
    rb.on_click(get_next)
    display(rb)
