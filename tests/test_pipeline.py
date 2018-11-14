import sys
sys.path.append("../")
import os

import unittest

import numpy as np
import numpy.random
import numpy.testing

from spotlob.spim import *
from spotlob.pipeline import *
import spotlob.process_opencv as p_cv
import spotlob.defaults as defaults


class TestSpimLifecycle(unittest.TestCase):
    def test_empty_spim_creation(self):
        s0 = Spim()

        def get_image(): return s0.image
        self.assertRaises(Exception, get_image)

        self.assertTrue(s0.stage == SpimStage.new)

        self.assertTrue(s0.cached == False)

    def test_reader(self):
        s0 = Spim()

        im_path = "testim.png"
        assert os.path.isfile(im_path)

        reader = p_cv.SimpleReader(im_path)
        s1 = s0.read(reader)

        self.assertTrue(s1.stage == SpimStage.loaded)

        try:
            im = s1.image
        except Exception:
            self.fail("could not load image")

        self.assertTrue(s1.image is np.ndarray)

    def test_portable_pipeline(self):
        mypipe = defaults.default_pipeline("testim.png")
        filename = "test_save.pipe"

        # store
        mypipe.save(filename)

        # restore
        loaded_pipe = mypipe.from_file(filename)

        # restored dicts have the same length
        self.assertTrue(len(loaded_pipe.process_stage_dict)
                        == len(mypipe.process_stage_dict))

        # # restored parameters are the same
        for restored, original_process in zip(loaded_pipe.process_stage_dict.values(), mypipe.process_stage_dict.values()):
            for p_r, p_o in zip(restored.parameters.parameters, original_process.parameters.parameters):
                self.assertTrue(p_r.name == p_o.name)
                self.assertTrue(p_r.value == p_o.value)
                self.assertTrue(p_r.type == p_o.type)

        # test processes are the same
        sample_grey_image = (np.random.rand(500, 500)*255).astype(np.uint8)

        preproc_mypipe = mypipe.process_stage_dict[SpimStage.converted].apply(
            sample_grey_image)
        preproc_loaded = mypipe.process_stage_dict[SpimStage.converted].apply(
            sample_grey_image)

        numpy.testing.assert_array_equal(preproc_mypipe, preproc_loaded)
