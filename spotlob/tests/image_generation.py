"""
This module provides functions to generate test images for spotlob tests
"""

import numpy as np


def binary_circle(shape=(1000, 1000), val_type=np.uint8, seed=None):
    """
    Creates a binary image with a circle that does not touch
    any border of the image

    PARAMS:
    -------
    shape: tuple of int
        (number of rows, number of columns) of the image in pixels
    val_type: numpy.dtype
        type of the values of the image array
    seed: int
        seed for numpy randomizer to get a fixed result

    RETURNS:
    --------
    tuple of image, position, radius
        image: numpy.array
            image which is zero everywhere except within a random circle where
            it has the maximum value of the dtype
        position: numpy.array
            position x and y of the circle
        radius: float
            radius of the circle
    """
    if seed is not None:
        np.random.seed(seed)

    # black image
    testim_shape = np.array(shape)
    testim = np.zeros(testim_shape).astype(val_type)
    testim_row_i, testim_col_i = np.indices(testim_shape)

    # random radius
    circ_r = np.random.random_sample()*min(testim_shape)/2.0

    # random position, but such that it does not touch border
    randfloats = np.random.random_sample(2)

    min_coord = np.zeros(2) + circ_r
    max_coord = testim_shape - circ_r

    circ_cy, circ_cx = min_coord + randfloats*(max_coord - min_coord)

    circle_mask = ((testim_row_i-circ_cy)**2 +
                   (testim_col_i-circ_cx)**2 <= circ_r**2)

    # maximum value of val_type within circle
    testim[circle_mask] = np.iinfo(val_type).max

    return testim, (circ_cx, circ_cy), circ_r
