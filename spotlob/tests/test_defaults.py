import unittest

from pkg_resources import resource_filename

from ..spim import Spim, SpimStage
from ..defaults import load_image


class TestDefaults(unittest.TestCase):
    def test_empty_spim_creation(self):
        image_filepath = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")

        myspim = load_image(image_filepath)
        self.assertTrue(myspim.stage == SpimStage.loaded)
