import unittest
from pkg_resources import resource_filename
import os.path

from ..output import Writer
from ..spim import Spim, SpimStage
from ..defaults import default_pipeline


class TestStorage(unittest.TestCase):
    def test_store_contour(self):
        mypipe = default_pipeline()
        filename = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")
        image_target_filename = resource_filename(
            "spotlob.tests", "resources/testdata3_spotlob.jpg")
        data_target_filename = resource_filename(
            "spotlob.tests", "resources/testdata3_spotlob.csv")
        s0 = Spim.from_file(filename, cached=True)

        s_final = mypipe.apply_all_steps(s0)

        writer = Writer(image_target_filename, data_target_filename)

        s_stored = s_final.store(writer)

        self.assertEqual(SpimStage.stored, s_stored.stage)

        self.assertTrue(os.path.isfile(data_target_filename))
        self.assertTrue(os.path.isfile(image_target_filename))

        # TODO: test contour on image
        # TODO: delete test files
