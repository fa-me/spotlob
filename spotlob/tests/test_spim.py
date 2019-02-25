import sys
import os
import unittest

from pkg_resources import resource_filename

import numpy as np
import numpy.random
from numpy.testing import assert_array_almost_equal,\
    assert_array_equal

from ..process_opencv import SimpleReader, \
    GreyscaleConverter, GaussianPreprocess, \
    BinaryThreshold
from ..defaults import default_pipeline
from ..spim import Spim, SpimStage


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
            s1im = s1.image
        except Exception:
            self.fail("could not load image")

        self.assertTrue(s1.stage == SpimStage.loaded)
        self.assertTupleEqual(s1im.shape, (2306, 2560, 3))

    def test_convert(self):
        reader = SimpleReader()
        converter = GreyscaleConverter()

        image_filepath = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")

        s2 = Spim.from_file(image_filepath)\
            .read(reader)\
            .convert(converter)

        self.assertTupleEqual(s2.image.shape, (2306, 2560))

    def test_preprocess(self):
        reader = SimpleReader()
        converter = GreyscaleConverter()
        preprocessor = GaussianPreprocess(ksize=5)

        image_filepath = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")

        s_final = Spim.from_file(image_filepath)\
            .read(reader)\
            .convert(converter)\
            .preprocess(preprocessor)

        self.assertTupleEqual(s_final.image.shape, (2306, 2560))

    def test_binarize(self):
        reader = SimpleReader()
        converter = GreyscaleConverter()
        preprocessor = GaussianPreprocess(ksize=5)
        binarizer = BinaryThreshold(100.0)

        image_filepath = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")

        s_final = Spim.from_file(image_filepath)\
            .read(reader)\
            .convert(converter)\
            .preprocess(preprocessor)\
            .binarize(binarizer)

        self.assertTupleEqual(s_final.image.shape, (2306, 2560))

        unique_values = np.sort(np.unique(s_final.image))
        assert_array_equal(unique_values, np.array([0, 255], dtype=np.uint8))

    def test_get_at_stage(self):
        reader = SimpleReader()
        converter = GreyscaleConverter()

        image_filepath = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")

        s0 = Spim.from_file(image_filepath, cached=True)
        s1 = s0.read(reader)
        s2 = s1.convert(converter)

        s0_ = s2.get_at_stage(SpimStage.new)
        s1_ = s2.get_at_stage(SpimStage.loaded)

        assert s0_ is s0
        assert s1_ is s1
