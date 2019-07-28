"""
This module provides a widget to browse through the
already applied detection, to see if everything worked
"""
import numpy as np

from ipywidgets import HBox, Button
from IPython.display import display, clear_output

from spotlob.preview import MatplotlibPreviewScreen
from spotlob.spim import Spim, SpimStage


class ReviewBrowser(object):
    index_state = 0
    crop_to_contour = False

    def __init__(self, dataframe, pipeline, crop_to_contour=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.screen = MatplotlibPreviewScreen()

        stage_dict = pipeline.process_stage_dict

        self.load_process = stage_dict[SpimStage.loaded-1]
        self.analysis_process = stage_dict[SpimStage.analyzed-1]
        self.crop_to_contour = crop_to_contour

    def show_next(self, btn):
        self.index_state = (self.index_state + 1) % len(self.dataframe)
        self.draw()

    def show_prev(self, btn):
        self.index_state = (self.index_state - 1) % len(self.dataframe)
        self.draw()

    def draw(self):
        row = self.dataframe.iloc[self.index_state, :]
        fp = row["filepath"]
        s0 = Spim.from_file(fp, cached=True).read(self.load_process)

        img = np.copy(s0.image)

        results_data = row.to_frame().T
        img_with_res = self.analysis_process.draw_results(img,
                                                          results_data,
                                                          self.crop_to_contour)

        if self.screen.fg_ax is None:
            self.screen.make_new(img_with_res)
        else:
            self.screen.update(img_with_res)

    def drop_current_row(self, btn):
        self.dataframe = self.dataframe.reset_index(drop=True) \
            .drop(self.index_state, axis=0)
        self.index_state = max(self.index_state-1, 0)
        self.draw()


def review_widget(dataframe, pipeline, crop_to_contour=False):
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
    browser = ReviewBrowser(dataframe, pipeline, crop_to_contour)

    next_button = Button(description="next")
    next_button.on_click(browser.show_next)

    drop_button = Button(description="drop")
    drop_button.on_click(browser.drop_current_row)

    prev_button = Button(description="prev")
    prev_button.on_click(browser.show_prev)

    display(HBox([prev_button, drop_button, next_button]))
    return browser
