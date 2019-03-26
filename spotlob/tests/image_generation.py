"""
This module provides functions to generate test images for spotlob tests
"""

import numpy as np
from ..calculation import distance_point_to_line


def binary_circle(center_pos,
                  radius,
                  shape=(1000, 1000),
                  val_type=np.uint8):
    """
    create a black image with a white circle in it

    PARAMS:
    -------
    center_pos: tuple of floats
        x and y position of the center of the circle
    radius: float
        radius of the circle
    shape: tuple of int
        (number of rows, number of columns) of the image in pixels
    val_type: numpy.dtype
        type of the values of the image array

    RETURNS:
    --------
    tuple of image, position, radius
        image: numpy.array
            image which is zero everywhere except within the given circle where
            it has the maximum value of the dtype
        position: numpy.array
            position x and y of the circle
        radius: float
            radius of the circle
    """
    # black image
    testim_shape = np.array(shape, dtype=np.uint16)
    testim = np.zeros(testim_shape).astype(val_type)
    testim_row_i, testim_col_i = np.indices(testim_shape)

    circ_cx, circ_cy = center_pos

    circle_mask = ((testim_row_i-circ_cy)**2 +
                   (testim_col_i-circ_cx)**2 <= radius**2)

    # maximum value of val_type within circle
    testim[circle_mask] = np.iinfo(val_type).max

    return testim, (circ_cx, circ_cy), radius


def binary_circle_off_border(shape=(1000, 1000), val_type=np.uint8, seed=None):
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

    testim_shape = np.array(shape)

    # random radius
    circ_r = np.random.random_sample()*min(testim_shape)/2.0

    # random position, but such that it does not touch border
    randfloats = np.random.random_sample(2)

    min_coord = np.zeros(2) + circ_r
    max_coord = testim_shape - circ_r

    circ_cy, circ_cx = min_coord + randfloats*(max_coord - min_coord)

    return binary_circle((circ_cx, circ_cy),
                         circ_r,
                         shape,
                         val_type)


def binary_circle_border(border,
                         shape=(1000, 1000),
                         val_type=np.uint8,
                         seed=None):
    """
    Creates a binary image with a circle that touches at least the left border
    of the image

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

    testim_shape = np.array(shape)

    # random radius, at most long side length of image/2
    circ_r = np.random.random_sample()*max(testim_shape)/2.0

    # random position, but such that it does touch one border
    randfloats = np.random.random_sample(2)

    if border == "left":
        circ_cx = randfloats[1]*circ_r
        circ_cy = randfloats[0]*testim_shape[0]
    elif border == "right":
        circ_cx = testim_shape[1] - randfloats[1]*circ_r
        circ_cy = randfloats[0]*testim_shape[0]
    elif border == "top":
        circ_cx = randfloats[1]*testim_shape[1]
        circ_cy = randfloats[0]*circ_r
    elif border == "bottom":
        circ_cx = randfloats[1]*testim_shape[1]
        circ_cy = testim_shape[0] - randfloats[0]*circ_r

    return binary_circle((circ_cx, circ_cy),
                         circ_r,
                         shape,
                         val_type)


def binary_line(position_A,
                position_B,
                width,
                shape=(1000, 1000),
                val_type=np.uint8):
    """
    create a black image with a white line in it

    PARAMS:
    -------
    position_A: tuple of int
        coordinates of position on the center of the line in px
    position_B: tuple of int
        coordinates of other position on the center of the line in px
    width: float
        width of the line in px
    shape: tuple of int
        (number of rows, number of columns) of the image in pixels
    val_type: numpy.dtype
        type of the values of the image array

    RETURNS:
    --------
    tuple of image, startposition, endposition, width
        image: numpy.array
            image which is zero everywhere except within a line,
            of given width, where it has the maximum value of the dtype
        width: float
            width of the line
    """
    # black image
    testim_shape = np.array(shape, dtype=np.uint16)
    testim = np.zeros(testim_shape).astype(val_type)
    j, i = np.indices(testim_shape)

    line_mask = distance_point_to_line(
        j, i, position_A, position_B) <= width/2.0

    # maximum value of val_type within linewidth
    testim[line_mask] = np.iinfo(val_type).max

    return testim, width
