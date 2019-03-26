import unittest
from pkg_resources import resource_filename
import os
import os.path

from ..output import Writer
from ..spim import Spim, SpimStage
from ..defaults import default_pipeline


class TestStorage(unittest.TestCase):

    temp_jpg_output = resource_filename("spotlob.tests",
                                        "resources/temp.jpg")
    temp_csv_output = resource_filename("spotlob.tests",
                                        "resources/temp.csv")

    def test_store_contour(self):

        filename = resource_filename(
            "spotlob.tests", "resources/testdata4.JPG")

        s0 = Spim.from_file(filename, cached=True)

        mypipe = default_pipeline()
        s_final = mypipe.apply_all_steps(s0)

        writer = Writer(self.temp_jpg_output,
                        self.temp_csv_output)
        s_stored = s_final.store(writer)

        self.assertEqual(SpimStage.stored, s_stored.stage)

        self.assertTrue(os.path.isfile(self.temp_csv_output))
        self.assertTrue(os.path.isfile(self.temp_jpg_output))

        # TODO: test contour on image

    def tearDown(self):
        os.remove(self.temp_jpg_output)
        os.remove(self.temp_csv_output)
