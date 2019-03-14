import unittest

import cv2
import numpy as np

from numpy.testing import assert_array_equal, assert_almost_equal

from .image_generation import binary_line

from ..spim import Spim, SpimStage
from ..process_opencv import ContourFinderSimple, FeatureFormFilter
from ..analyse_line import LineAnalysis
from ..calculation import points_within_contours


class LineDetectionTestCase(unittest.TestCase):
    seed = 0
    repetitions = 1

    def test_empty_image(self):
        h, w = [1000, 2000]

        contour_finder = ContourFinderSimple()
        feature_filter = FeatureFormFilter(
            size=0, solidity=0.9, remove_on_edge=False)
        line_analysis = LineAnalysis(linewidth_percentile=99)

        black = np.zeros((h, w), dtype=np.uint8)

        bin_spim = Spim(image=black,
                        metadata={},
                        stage=SpimStage.binarized,
                        cached=False,
                        predecessors=[])

        res_spim = bin_spim\
            .extract_features(contour_finder)\
            .filter_features(feature_filter)\
            .analyse(line_analysis)

        res_df = res_spim.metadata["results"]

        assert len(res_df) == 0

    def test_binary_line_detection(self):
        h, w = [1000, 2000]

        contour_finder = ContourFinderSimple()
        feature_filter = FeatureFormFilter(
            size=0, solidity=0.9, remove_on_edge=False)
        line_analysis = LineAnalysis(linewidth_percentile=99)

        np.random.seed(self.seed)

        for i in range(self.repetitions):
            x1, x2 = np.random.randint(0, w, size=2)
            y1, y2 = np.random.randint(0, h, size=2)
            posA = (x1, y1)
            posB = (x2, y2)

            width = np.random.random_sample()*min((w, h))/10

            # create a random line
            line_im, linewidth = binary_line(posA, posB, width)

            bin_spim = Spim(image=line_im,
                            metadata={},
                            stage=SpimStage.binarized,
                            cached=False,
                            predecessors=[])

            res_spim = bin_spim\
                .extract_features(contour_finder)\
                .filter_features(feature_filter)\
                .analyse(line_analysis)

            res_df = res_spim.metadata["results"]

            result_width_percentile = res_df.loc[0, "linewidth_px"]
            result_width_bb = res_df.loc[0, "bb_width_px"]
            result_width_area = res_df.loc[0, "linewidth2_px"]

            # compare analysis with input width
            assert_almost_equal(result_width_percentile, linewidth, decimal=0)
            assert_almost_equal(result_width_bb, linewidth, decimal=1)
            assert_almost_equal(result_width_area, linewidth, decimal=0)

    # def test_contour_mask(self):
    #     # create an image with a line
    #     im, _ = binary_line((100, 100),
    #                         (300, 400),
    #                         150,
    #                         shape=(2000, 2000))

    #     # find contours
    #     _, contours, _ = cv2.findContours(
    #         im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # get points within contours
    #     detected_points = points_within_contours(contours)

    #     # check if difference towards original image is zero
    #     im_J, im_I = np.indices(im.shape)
    #     im_mask = im.astype(bool)

    #     original_points = np.vstack([im_I[im_mask], im_J[im_mask]]).T

    #     assert_array_equal(original_points, detected_points)
