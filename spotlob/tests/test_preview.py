import unittest

from pkg_resources import resource_filename
import numpy as np
import numpy.testing

from ..defaults import default_pipeline
from ..spim import Spim, SpimStage
from ..preview import MatplotlibPreviewScreen


class TestPreview(unittest.TestCase):
    def test_preview_default_pipe(self):
        mypipe = default_pipeline()
        filename = resource_filename(
            "spotlob.tests", "resources/testdata4.JPG")
        s0 = Spim.from_file(filename, cached=True)
        s_final = mypipe.apply_all_steps(s0)

        input_stages_without_preview = [
            SpimStage.new
        ]

        for input_stage, process in mypipe.process_stage_dict.items():

            def get_preview():
                return process.preview(s_final)

            if input_stage in input_stages_without_preview:
                self.assertRaises(NotImplementedError, get_preview)
            else:
                self.assertEqual(
                    s_final.image.shape[:2], get_preview().shape[:2])

    def test_mpl_previewscreen(self):
        preview_screen = MatplotlibPreviewScreen()

        mypipe = default_pipeline()

        filename = resource_filename(
            "spotlob.tests", "resources/testdata4.JPG")
        s0 = Spim.from_file(filename, cached=True)
        s_final = mypipe.apply_all_steps(s0)

        input_stages_without_preview = [
            SpimStage.new,
            SpimStage.postprocessed
        ]

        preview_screen.make_new(s_final.image)

        for input_stage, process in mypipe.process_stage_dict.items():
            if input_stage not in input_stages_without_preview:
                preview_image = process.preview(s_final)
                preview_screen.update(preview_image)
