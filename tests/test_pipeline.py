import sys
sys.path.append("../")
import os

import unittest

import numpy as np

from spotlob.spim import *
from spotlob.pipeline import *
import spotlob.process_opencv as p_cv


class TestSpimLifecycle(unittest.TestCase):
    def test_empty_spim_creation(self):
        s0 = Spim()

        def get_image(): return s0.image
        self.assertRaises(Exception, get_image)

        self.assertTrue(s0.stage == SpimStage.new)

        self.assertTrue(s0.cached == False)

    def test_reader(self):
        s0 = Spim()

        im_path = os.path.abspath("Bild--02.jpg")
        assert os.path.isfile(im_path)

        reader = p_cv.SimpleReader(im_path)
        s1 = s0.read(reader)

        self.assertTrue(s1.stage == SpimStage.loaded)

        try:
            im = s1.image
        except Exception:
            self.fail("could not load image")

        self.assertTrue(s1.image is np.ndarray)
