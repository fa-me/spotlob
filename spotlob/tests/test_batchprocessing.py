import unittest
import os
from pkg_resources import resource_filename
from pandas.testing import assert_frame_equal

from ..batch import batchprocess
from ..defaults import default_pipeline
from ..spim import SpimStage


class BatchProcessingTestCase(unittest.TestCase):
    temp_pipe_filename = resource_filename("spotlob.tests",
                                           "resources/temp.pipe")

    def setUp(self):
        mypipe = default_pipeline()

        # adjust filter criteria
        filterprocess = mypipe.process_stage_dict[SpimStage.features_extracted]
        filterprocess.parameters["minimal_area"].value = 500
        filterprocess.parameters["solidity_limit"].value = 0.5

        mypipe.save(self.temp_pipe_filename)

    def test_batchprocess_small_multiprocessing(self):
        small_batch_f = ["testdata4.JPG", "testdata5.JPG"]
        small_batch = [resource_filename("spotlob.tests",
                                         os.path.join("resources/", im_file))
                       for im_file in small_batch_f]

        results_no_mp = batchprocess(self.temp_pipe_filename,
                                     small_batch,
                                     multiprocessing=False)
        results_mp = batchprocess(self.temp_pipe_filename,
                                  small_batch,
                                  multiprocessing=True)
        assert len(results_mp) > 0
        assert_frame_equal(results_mp, results_no_mp)

        self.assertEqual(len(results_mp), 20+22)

    def tearDown(self):
        os.remove(self.temp_pipe_filename)
