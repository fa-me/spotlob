import unittest

from pkg_resources import resource_filename

from ..defaults import make_gui, show_gui


class TestGUI(unittest.TestCase):
    def test_gui_creation(self):
        image_filepath = resource_filename(
            "spotlob.tests", "resources/testdata3.jpg")

        gui = make_gui(image_filepath)

        show_gui(gui)
