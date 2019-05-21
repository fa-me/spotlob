import unittest

from pkg_resources import resource_filename
import numpy as np
import numpy.testing

from ..defaults import default_pipeline
from ..spim import Spim, SpimStage
from ..review import ReviewBrowser, review_widgets
from ..batch import batchprocess


class TestReview(unittest.TestCase):
    def setUp(self):
        test_images = [resource_filename("spotlob.tests",
                                         "resources/testdata4.JPG"),
                       resource_filename("spotlob.tests",
                                         "resources/testdata5.JPG")]
        mypipe = default_pipeline()
        pipe_filename = resource_filename("spotlob.tests",
                                          "resources/test_save.pipe")
        mypipe.save(pipe_filename)

        self.dummy_pipeline = mypipe
        self.dummy_dataset = batchprocess(pipe_filename, test_images)

    def test_browser_increment(self):
        browser = ReviewBrowser(self.dummy_dataset,
                                self.dummy_pipeline)

        assert browser.index_state == 0
        browser.show_next(None)
        assert browser.index_state == 1
        browser.show_prev(None)
        assert browser.index_state == 0
        browser.show_prev(None)
        assert browser.index_state > 0

    def test_browser_drop_row(self):
        browser = ReviewBrowser(self.dummy_dataset,
                                self.dummy_pipeline)
        browser.drop_current_row(None)

        assert len(browser.dataframe) == len(self.dummy_dataset) - 1
