import unittest

import cv2
import numpy as np

from numpy.testing import assert_array_equal

from .image_generation import binary_line

from ..spim import Spim, SpimStage
from ..process_opencv import ContourFinderSimple, FeatureFormFilter
from ..analyse_line import LineAnalysis
from ..calculation import points_within_contours


class LineDetectionTestCase(unittest.TestCase):
    seed = 0
    repetitions = 10

    def test_binary_line_detection(self):
        h, w = [1000, 2000]

        contour_finder = ContourFinderSimple()
        feature_filter = FeatureFormFilter(
            size=0, solidity=0.9, remove_on_edge=False)
        line_analysis = LineAnalysis()

        for i in range(self.repetitions):
            # TODO: create a random line
            # TODO: analyze
            # TODO: compare analysis with input width
            self.fail()

    def test_contour_mask(self):
        # create an image with a line
        im, _ = binary_line((100, 100),
                            (300, 400),
                            150,
                            shape=(2000, 2000))

        # find contours
        _, contours, _ = cv2.findContours(
            im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get points within contours
        detected_points = points_within_contours(contours)

        # check if difference towards original image is zero
        im_J, im_I = np.indices(im.shape)
        im_mask = im.astype(bool)

        original_points = np.vstack([im_I[im_mask], im_J[im_mask]]).T

        assert_array_equal(original_points, detected_points)
