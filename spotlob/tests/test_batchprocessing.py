import unittest
import os
from pkg_resources import resource_filename
from numpy.testing import assert_array_equal

from ..batch import batchprocess
from ..defaults import default_pipeline


class BatchProcessingTestCase(unittest.TestCase):
    temp_pipe_filename = resource_filename("spotlob.tests",
                                           "resources/test_default.pipe")

    def setUp(self):
        mypipe = default_pipeline()
        mypipe.save(self.temp_pipe_filename)

    def test_batchprocess_small_multiprocessing(self):
        small_batch = ["temp.tif", "temp1.tif", "testdata3.jpg"]
        small_batch = [resource_filename("spotlob.tests",
                                         os.path.join("resources/", im_file))
                       for im_file in small_batch]

        results_no_mp = batchprocess(self.temp_pipe_filename,
                                     small_batch,
                                     multiprocessing=False)
        results_mp = batchprocess(self.temp_pipe_filename,
                                  small_batch,
                                  multiprocessing=True)

        assert_array_equal(results_mp, results_no_mp)

    def tearDown(self):
        os.remove(self.temp_pipe_filename)
