import unittest
from pkg_resources import resource_filename

import numpy as np
from numpy.testing import assert_array_equal,\
    assert_array_almost_equal, assert_almost_equal

from .image_generation import binary_circle_off_border

from ..spim import Spim, SpimStage
from ..process_opencv import ContourFinderSimple, FeatureFormFilter
from ..output import Writer
from ..analyze_circle import CircleAnalysis
from ..defaults import default_pipeline


class CircleDetectionTestCase(unittest.TestCase):
    seed = 0
    repetitions = 10

    def test_binary_artificial_circle_detection(self):
        h, w = [1000, 2000]

        contour_finder = ContourFinderSimple()
        feature_filter = FeatureFormFilter(
            size=0, solidity=0.9, remove_on_edge=True)
        circle_analysis = CircleAnalysis()

        for i in range(self.repetitions):
            # create 8bit binary image with circle
            circ_im, exp_pos, exp_radius = binary_circle_off_border(
                shape=(h, w),
                val_type=np.uint8,
                seed=self.seed)

            assert_array_equal(np.sort(np.unique(circ_im)), np.array([0, 255]))

            # make spim, assuming image is already binary
            bin_spim = Spim(image=circ_im,
                            metadata={},
                            stage=SpimStage.binarized,
                            cached=False,
                            predecessors=[])

            cont_spim = bin_spim\
                .extract_features(contour_finder)

            res_spim = cont_spim\
                .filter_features(feature_filter)\
                .analyze(circle_analysis)

            res_df = res_spim.metadata["results"]

            res_position = res_df.loc[0, "ellipse_position_px"]
            res_MA = res_df.loc[0, "ellipse_majorAxis_px"]
            res_ma = res_df.loc[0, "ellipse_minorAxis_px"]

            assert_array_almost_equal(
                np.array(res_position), exp_pos, decimal=1)
            assert_almost_equal(res_MA/2, exp_radius, decimal=0)
            assert_almost_equal(res_ma/2, exp_radius, decimal=0)

    def test_number_detected_spots_testdata4(self):
        filename = resource_filename("spotlob.tests",
                                     "resources/testdata4.JPG")
        s0 = Spim.from_file(filename, cached="True")

        mypipe = default_pipeline()

        filterprocess = mypipe.process_stage_dict[SpimStage.features_extracted]
        filterprocess.parameters["minimal_area"].value = 500
        filterprocess.parameters["solidity_limit"].value = 0.5

        s_final = mypipe.apply_all_steps(s0)

        data = s_final.get_data()

        # wr = Writer(resource_filename("spotlob.tests",
        #                               "resources/testdata4_detected.JPG"),
        #             resource_filename("spotlob.tests",
        #                               "resources/testdata4_detected.csv"))

        # s_final.store(wr)

        # expect 22 spots
        self.assertEqual(len(data), 22)

    def test_number_detected_spots_testdata5(self):
        filename = resource_filename("spotlob.tests",
                                     "resources/testdata5.JPG")
        s0 = Spim.from_file(filename, cached="True")

        mypipe = default_pipeline()

        filterprocess = mypipe.process_stage_dict[SpimStage.features_extracted]
        filterprocess.parameters["minimal_area"].value = 500
        filterprocess.parameters["solidity_limit"].value = 0.5

        s_final = mypipe.apply_all_steps(s0)

        data = s_final.get_data()

        # expect 20 spots
        self.assertEqual(len(data), 20)

    def test_empty_file(self):
        filename = resource_filename("spotlob.tests",
                                     "resources/testdata6.JPG")
        s0 = Spim.from_file(filename)

        mypipe = default_pipeline(thresholding="simple")

        s_final = mypipe.apply_all_steps(s0)

        data = s_final.get_data()

        # expect 20 spots
        self.assertEqual(len(data), 0)
