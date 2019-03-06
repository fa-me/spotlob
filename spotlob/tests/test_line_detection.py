import unittest

from .image_generation import binary_line

from ..spim import Spim, SpimStage
from ..process_opencv import ContourFinderSimple, FeatureFormFilter
from ..analyse_line import LineAnalysis


class LineDetectionTestCase(unittest.TestCase):
    seed = 0
    repetitions = 10

    def test_binary_circle_detection(self):
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
