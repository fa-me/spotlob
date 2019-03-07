import numpy as np
import cv2


def distance_point_to_line(x0, y0, pt_on_line1, pt_on_line2):
    """
    Distance from point (x0, y0) to a line defined by two points
    pt_on_line1 and pt_on_line2
    see https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points # noqa
    """

    x1, y1 = pt_on_line1
    x2, y2 = pt_on_line2

    numerator = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denom = np.sqrt((y2-y1)**2 + (x2-x1)**2)

    return numerator / denom


def max_extends(contours):
    max_extends = np.array([c.max(axis=0) for c in contours])
    return max_extends.max(axis=0)[0]


def points_within_contours(contours):
    """
    gives the indices of points within the given contours
    using the drawContours function
    """

    extend_x, extend_y = max_extends(contours)

    # get maximum points from contours to create mask image
    # of that shape that just covers all

    mask = np.zeros((extend_y, extend_x))

    # fill mask image
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)

    mask = mask.astype(bool)

    yi, xi = np.indices((extend_y, extend_x))
    return np.vstack([xi[mask], yi[mask]]).T
