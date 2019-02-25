import unittest

import numpy as np
from numpy.testing import assert_array_equal,\
    assert_array_almost_equal, assert_almost_equal

from .image_generation import binary_circle_border

from ..spim import Spim, SpimStage
from ..process_opencv import ContourFinderSimple, FeatureFormFilter


class FeatureFilterTestCase(unittest.TestCase):
    seed = 0
    repetitions = 20

    def test_binary_circle_left_border_filter(self):
        h, w = [1000, 2000]

        contour_finder = ContourFinderSimple()
        feature_filter = FeatureFormFilter(size=0, solidity=0.9)

        for i in range(self.repetitions):
            j = np.random.randint(low=0, high=3)
            border = ["left", "right", "top", "bottom"][j]

            circ_im, exp_pos, exp_radius = binary_circle_border(
                border,
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
                .extract_features(contour_finder)\
                .filter_features(feature_filter)

            blobs = cont_spim.metadata["contours"]
            self.assertEqual(len(blobs), 0)
