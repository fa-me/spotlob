import unittest

import pandas as pd
import numpy

from .image_generation import binary_circle_off_border
from ..defaults import default_pipeline
from ..register import PROCESS_REGISTER as preg
from ..spim import Spim, SpimStage
from ..process_steps import Analysis


class TestProcessRegister(unittest.TestCase):
    def setUp(self):
        self.test_pipe = default_pipeline()

        im, _, __ = binary_circle_off_border()
        self.test_spim = Spim(im,
                              metadata={},
                              stage=SpimStage.binarized,
                              cached=True,
                              predecessors={})

    def test_register_analysis_plugin(self):

        def my_analysis(contours):
            return pd.DataFrame([{"number_of_contours": len(contours)}], index=[0])

        # register class
        preg.analysis_plugin([])(my_analysis)

        # check if function name is in register
        self.assertTrue(my_analysis.__name__ in preg.available_processes)

        self.test_pipe.process_stage_dict[SpimStage.features_filtered] = \
            preg.available_processes[my_analysis.__name__]

        results = self.test_pipe.apply_at_stage(
            self.test_spim).metadata["results"]

        self.assertTrue("number_of_contours" in results.columns)
