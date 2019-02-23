import sys
import os
import unittest

from pkg_resources import resource_filename

import numpy as np
import numpy.random
import numpy.testing

from ..spim import *
from ..pipeline import *
from ..process_opencv import SimpleReader
from ..defaults import default_pipeline


class TestSpimLifecycle(unittest.TestCase):
    def test_empty_spim_creation(self):
        image_filepath = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")

        s0 = Spim.from_file(image_filepath)

        def get_image():
            return s0.image

        self.assertRaises(Exception, get_image)
        self.assertTrue(s0.stage == SpimStage.new)
        self.assertTrue(s0.cached is False)

    def test_reader(self):
        image_filepath = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")
        s0 = Spim.from_file(image_filepath)

        reader = SimpleReader()

        try:
            s1 = s0.read(reader)
            im = s1.image
        except Exception:
            self.fail("could not load image")

        self.assertTrue(s1.stage == SpimStage.loaded)
        self.assertTupleEqual(s1.image.shape, (2306, 2560, 3))

    def test_portable_pipeline(self):
        mypipe = default_pipeline()
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
